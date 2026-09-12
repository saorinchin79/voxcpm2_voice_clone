#!/usr/bin/env python3
"""Manage the VoxCPM2 Studio login accounts stored in users.json.

Passwords are always typed at a prompt, never passed on the command line:
argv ends up in the shell history and in `ps` output for every user on the box.
"""
import argparse
import getpass
import sys
import time

import auth

MIN_PASSWORD_LEN   = 6
MAX_PASSWORD_TRIES = 3


def _fmt_time(ts):
    if not ts:
        return '-'
    try:
        return time.strftime('%Y-%m-%d %H:%M', time.localtime(float(ts)))
    except (TypeError, ValueError, OSError):
        return '-'


def _prompt_new_password(username):
    """Ask twice for a password, re-prompting on an empty/short/mismatched entry."""
    print(f"Setting the password for '{username}'. "
          f'Minimum {MIN_PASSWORD_LEN} characters, nothing is echoed.', flush=True)
    for remaining in range(MAX_PASSWORD_TRIES - 1, -1, -1):
        first  = getpass.getpass('New password: ')
        second = getpass.getpass('Repeat password: ')

        if not first:
            problem = 'password must not be empty'
        elif len(first) < MIN_PASSWORD_LEN:
            problem = f'password must be at least {MIN_PASSWORD_LEN} characters'
        elif first != second:
            problem = 'the two entries did not match'
        else:
            return first

        if remaining:
            print(f'{problem} - try again ({remaining} attempt(s) left).', file=sys.stderr)
        else:
            raise auth.AuthError(f'{problem} - giving up after '
                                 f'{MAX_PASSWORD_TRIES} attempts, nothing was changed')


def _drop_sessions(username):
    """Kill live logins so an old password cannot keep an existing browser signed in."""
    dropped = auth.SessionStore().destroy_user(username)
    if dropped:
        print(f"Dropped {dropped} active session(s) for '{username}'.")
    else:
        print(f"No active sessions for '{username}' to drop.")


def _confirm_delete(username):
    print(f"About to delete '{username}'. This cannot be undone.")
    typed = input('Type the username to confirm (or anything else to cancel): ').strip()
    return typed == username


def cmd_list(args):
    users = auth.UserStore().list_users()
    if not users:
        print('No users yet. Run "python manage_users.py seed" to create the defaults.')
        return 0

    rows = [(u['username'], u.get('role', ''),
             _fmt_time(u.get('created')), _fmt_time(u.get('last_login')))
            for u in sorted(users, key=lambda u: u['username'])]
    headers = ('USERNAME', 'ROLE', 'CREATED', 'LAST LOGIN')
    widths  = [max(len(str(r[i])) for r in (headers,) + tuple(rows)) for i in range(4)]

    def line(cells):
        return '  '.join(str(c).ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    print(line(headers))
    print(line(['-' * w for w in widths]))
    for row in rows:
        print(line(row))
    print(f'\n{len(rows)} user(s).')
    return 0


def cmd_add(args):
    store = auth.UserStore()
    if store.get(args.username):
        raise auth.AuthError(f"user '{args.username}' already exists "
                             f'(use "passwd" to change the password)')
    password = _prompt_new_password(args.username)
    user = store.add_user(args.username, password, role=args.role)
    print(f"Created '{user['username']}' with role {user['role']}.")
    _drop_sessions(user['username'])
    return 0


def cmd_passwd(args):
    store = auth.UserStore()
    existing = store.get(args.username)
    if not existing:
        raise auth.AuthError(f"no such user: '{args.username}'")
    password = _prompt_new_password(existing['username'])
    user = store.set_password(existing['username'], password)
    print(f"Password updated for '{user['username']}'.")
    _drop_sessions(user['username'])
    return 0


def cmd_delete(args):
    store = auth.UserStore()
    existing = store.get(args.username)
    if not existing:
        raise auth.AuthError(f"no such user: '{args.username}'")
    name = existing['username']
    if not args.yes and not _confirm_delete(name):
        print('Cancelled, nothing was deleted.', file=sys.stderr)
        return 1
    store.delete_user(name)
    print(f"Deleted '{name}'.")
    _drop_sessions(name)
    return 0


def cmd_seed(args):
    created = auth.UserStore().seed_defaults()
    if created:
        print('Created: ' + ', '.join(created))
    else:
        print('Nothing to do, every default user already exists.')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog='manage_users.py',
        description='Manage the VoxCPM2 Studio login accounts (users.json).',
        epilog='examples:\n'
               '  python manage_users.py seed              create the default accounts on a fresh install\n'
               '  python manage_users.py list              show every account and when it last signed in\n'
               '  python manage_users.py passwd admin      rotate a password (prompts, then kills live sessions)\n'
               '  python manage_users.py add chhay         add an account, password is prompted twice\n',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', metavar='{list,add,passwd,delete,seed}',
                                required=True)

    p = sub.add_parser('list', help='list accounts with role, created and last login')
    p.set_defaults(func=cmd_list)

    p = sub.add_parser('add', help='add an account (password is prompted, never argv)')
    p.add_argument('username')
    p.add_argument('--role', default='admin', help='account role (default: admin)')
    p.set_defaults(func=cmd_add)

    p = sub.add_parser('passwd', help='change an account password')
    p.add_argument('username')
    p.set_defaults(func=cmd_passwd)

    p = sub.add_parser('delete', help='delete an account')
    p.add_argument('username')
    p.add_argument('--yes', action='store_true', help='skip the typed confirmation')
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser('seed', help='create any missing default accounts (safe to repeat)')
    p.set_defaults(func=cmd_seed)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except auth.AuthError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print('\naborted, nothing was changed', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
