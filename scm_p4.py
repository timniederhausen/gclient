import os
import re
import subprocess
import marshal
import io
import socket
import logging
import zlib
import typing
import urllib.parse as urlparse
from collections import namedtuple

PerforceWorkspaceUrl = namedtuple(
    'PerforceWorkspaceUrl', 'port username password path depot_type changelist')


# format: p4://<p4port>/<depot path or stream path>[?type=stream][&cl=changelist]
def parse_workspace_url(url: str):
    components = urlparse.urlparse(url, allow_fragments=True)
    assert components.scheme == 'p4'

    if components.port:
        p4port = components.hostname + ':' + str(components.port)
    else:
        p4port = components.hostname

    query_params = urlparse.parse_qs(components.query)

    changelist = None
    if 'cl' in query_params:
        changelist = int(query_params['cl'][0])

    depot_type = 'local'
    if 'type' in query_params:
        depot_type = query_params['type'][0]

    return PerforceWorkspaceUrl(p4port, components.username,
                                components.password, '//' + components.path[1:],
                                depot_type, changelist)


def unmarshal_p4_output(stdout: bytes, check=False):
    # https://www.perforce.com/manuals/p4api.net/p4api.net_reference/html/T_Perforce_P4_ErrorSeverity.htm
    E_FAILED = 3

    fp = io.BytesIO(stdout)
    results: typing.List[typing.Dict[bytes, bytes]] = []
    try:
        while True:
            result = marshal.load(fp)
            if check and result[b'code'] == b'error' and result[
                    b'severity'] >= E_FAILED:
                raise Exception(
                    f'Failed P4 call: {result.get(b"data", result)}')
            results.append(result)
    except EOFError:
        pass
    return results


def format_marshalled_command(cmdargs: typing.List[str]):
    return ['p4', '-G', '-ztag'] + cmdargs


def _encode(o):
    if isinstance(o, str):
        return o.encode('utf-8')
    if isinstance(o, bytes):
        return o
    return bytes(o)


def encode_marshalled_command_input(input: typing.Dict[typing.AnyStr,
                                                       typing.AnyStr]):
    return marshal.dumps({_encode(k): v for k, v in input.items()}, 0)


def format_client_name(checkout_path: str, parsed_url: PerforceWorkspaceUrl):
    try:
        client_name_template = os.environ['P4CLIENT_TEMPLATE']
    except KeyError:
        client_name_template = "{user}_{host}_{depot_path_as_name}_{suffix}"
    return client_name_template.format(
        user=parsed_url.username or os.environ.get('P4USER'),
        host=socket.getfqdn().partition('.')[0],
        depot_path_as_name=parsed_url.path[2:].replace('/', '_'),
        suffix=zlib.crc32(checkout_path.encode('utf-8')))


class P4Environment(object):

    def __init__(self,
                 port: str = None,
                 user: str = None,
                 passwd: str = None,
                 client: str = None):
        self.port = port
        self.user = user
        self.passwd = passwd
        self.client = client

    @staticmethod
    def from_workspace(checkout_path: str):
        try:
            output = subprocess.check_output(
                ['p4', 'set', '-q'], cwd=checkout_path,
                universal_newlines=True).split('\n')
        except subprocess.CalledProcessError:
            return P4Environment()
        except FileNotFoundError: # cwd does not exist
            return P4Environment()

        if not output:
            raise RuntimeError(f'Empty response from p4 set')

        records = {}
        for line in output:
            tokens = line.partition('=')
            records[tokens[0]] = tokens[2]

        return P4Environment(records.get('P4PORT'), records.get('P4USER'),
                             records.get('P4PASSWD'), records.get('P4CLIENT'))


class P4Context(P4Environment):

    def prepare_environment(self, kwargs):
        env = kwargs.pop('env', None) or os.environ.copy()
        if self.port:
            env['P4PORT'] = self.port
        if self.user:
            env['P4USER'] = self.user
        if self.passwd:
            env['P4PASSWD'] = self.passwd
        if self.client:
            env['P4CLIENT'] = self.client
        return env

    def call(self, cmdargs: typing.List[str], **kwargs):
        return subprocess.run(['p4'] + cmdargs,
                              text=True,
                              env=self.prepare_environment(kwargs),
                              **kwargs)

    def run(self, cmdargs: typing.List[str], **kwargs):
        proc = subprocess.run(['p4'] + cmdargs,
                              stdout=subprocess.PIPE,
                              text=True,
                              env=self.prepare_environment(kwargs),
                              **kwargs)
        return proc.stdout

    def run_marshalled(self,
                       cmdargs: typing.List[str],
                       input: dict = None,
                       check=False,
                       **kwargs):
        if input:
            dumped_input = encode_marshalled_command_input(input)
        else:
            dumped_input = None

        args = format_marshalled_command(cmdargs)
        env = self.prepare_environment(kwargs)
        logging.debug('P4: \'%s\' for \'%s\'', ' '.join(args),
                      env.get('P4CLIENT', 'none'))

        # For -G: https://community.perforce.com/s/article/3518
        proc = subprocess.run(args,
                              input=dumped_input,
                              stdout=subprocess.PIPE,
                              env=env,
                              **kwargs)
        output = unmarshal_p4_output(proc.stdout, check)
        if check and 0 != proc.returncode:
            raise Exception(f'Failed to run {str(args)}: {str(output)}')
        return output

    @staticmethod
    def for_checkout(checkout_path: str, url: str, client_name: str = None):
        parsed_url = parse_workspace_url(url)
        default_environment = P4Environment.from_workspace(checkout_path)
        user = parsed_url.username or default_environment.user
        password = parsed_url.password or os.environ.get('P4PASSWD')
        if client_name is None:
            client_name = default_environment.client
        if client_name is None:
            client_name = format_client_name(checkout_path, parsed_url)
        return P4Context(parsed_url.port, user, password, client_name)


def _has_whitespace(*args: str):
    return any([re.search(r'\s', i) for i in args if i is not None])


def make_client_spec(ctx: P4Context,
                     name: str,
                     root: str,
                     owner: str = None,
                     host: str = None,
                     description: str = None,
                     options: typing.List[str] = None,
                     submit_options: typing.List[str] = None,
                     eol='local',
                     stream: str = None,
                     views: typing.Dict[typing.AnyStr, typing.AnyStr] = None):
    if submit_options is None:
        submit_options = ['SubmitUnchanged']
    if options is None:
        options = []
    if not host:
        host = socket.getfqdn().partition('.')[0]
    if not owner:
        owner = ctx.user
    if description is None:
        description = 'Created by ' + owner

    spec = dict(
        Client=name,
        Owner=owner,
        Description=description,
        Host=host,
        Root=os.path.abspath(root),
        Options=' '.join(options),
        SubmitOptions='+'.join(submit_options),
        LineEnd=eol,
    )
    if stream:
        spec['Stream'] = stream
    else:
        if views is None:
            views = [('//...', '...')]
        for i, (k, v) in enumerate(views):
            qa = '"' if _has_whitespace(k) else ''
            qb = '"' if _has_whitespace(name, v) else ''
            spec[f'View{i}'] = f'{qa}{k} {qb}//{name}/{v}{qb}\n'
    return spec


def create_client(ctx: P4Context,
                  name: str,
                  root: str,
                  owner: str = None,
                  host: str = None,
                  description: str = None,
                  options: typing.List[str] = None,
                  submit_options: typing.List[str] = None,
                  eol='local',
                  stream: str = None,
                  views: typing.Dict[typing.AnyStr, typing.AnyStr] = None):
    spec = make_client_spec(ctx, name, root, owner, host, description, options,
                            submit_options, eol, stream, views)
    return ctx.run_marshalled(['client', '-i'], input=spec, check=True)


def create_change(ctx: P4Context,
                  description: str = '',
                  type: str = None,
                  user: str = None):
    description = description.replace('\n', '\n\t')
    spec = dict(Change=b'new', Description=description)
    if type:
        spec['Type'] = type
    if user:
        spec['User'] = user or ctx.user

    output = ctx.run_marshalled(['change', '-i'], input=spec)
    for result in output:
        mo = re.search(b'Change (.+) created.$', result[b'data'], re.M)
        if mo:
            return int(mo.group(1))

    raise Exception('Failed to create change: ' + str(output))


def stat_change(ctx: P4Context, cl: int) -> dict:
    output = ctx.run_marshalled(['change', '-o', str(cl)])
    for result in output:
        if result[b'code'] == b'stat':
            return result
    raise Exception('Failed to stat change: ' + str(output))


def delete_change(ctx: P4Context, cl: int) -> bool:
    output = ctx.run(['change', '-d', str(cl)])
    if re.search(r'Change (.+) deleted.', output,
                 re.M) and 'can\'t be deleted.' not in output:
        return True
    return False


_RETRY_SUBMIT_ERRORS = (
    " - must resolve",
    " - already locked by",
    " - add of added file",
    " - edit of deleted file",
)


def submit_change(ctx: P4Context, cl: int):
    for _ in range(40):
        info = stat_change(ctx, cl)

        state = info[b'Status']
        if state != b'pending':
            raise Exception(f'Change {cl} is in invalid state {state}')

        has_files = any((k.startswith(b'Files') for k in info.keys()))
        if not has_files:
            raise Exception(f'Change {cl} has no files')

        try:
            output = ctx.run(['submit', '-c',
                              str(cl), '-f', 'submitunchanged'],
                             check=True)
            output = output.strip()
            if output.endswith('submitted.'):
                new_cl = 0
                if output.endswith('and submitted.'):
                    mo = re.search(r'renamed change (.+) and submitted.$',
                                   output, re.M)
                    new_cl = int(mo.group(1))
                else:
                    mo = re.search(r'Change (.+) submitted.$', output, re.M)
                    new_cl = int(mo.group(1))
                if new_cl < cl:
                    raise Exception(
                        f'Failed to get change number from submitted: {output}')
                return new_cl
        except subprocess.CalledProcessError as e:
            print(e.output)
            done = set()
            for line in e.output.split():  # type: str
                for known_err in _RETRY_SUBMIT_ERRORS:
                    path_start = line.find('//')
                    error_start = line.find(known_err, path_start + 1)
                    if path_start == -1 or error_start == -1:
                        continue
                    filename = line[path_start:error_start]
                    if filename.find('//') != filename.rfind('//'):
                        continue
                    if filename in done:
                        continue
                    print(f'Brutal resolve on {filename}')
                    ctx.run(['revert', '-c', str(cl), '-k', filename])
                    ctx.run(['sync', '-f', '-k', f'{filename}#head'])
                    ctx.run(['reconcile', '-c', str(cl), '-ea'])
                    done.add(filename)

    raise Exception(f'Giving up on changelist {cl}')
