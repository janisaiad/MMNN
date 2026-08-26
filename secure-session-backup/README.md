# Encrypted session archive

This directory contains an encrypted Markdown export of the machine's AI sessions.
The original session files were not deleted or modified.

- Snapshot date: 2026-08-26 UTC
- Session Markdown files: 31
- Exported messages: 45,628
- Archive encryption: GnuPG symmetric AES-256
- Compression: Zstandard
- SHA-256: `1fd5153c6c0adbe39f91f22eb96c38a53eec8f64413ef0499f99b174603a8393`

The passphrase is intentionally not stored in this repository. Run:

```bash
./secure-session-backup/restore.sh \
  ./secure-session-backup/sessions-20260826.tar.zst.gpg \
  ./restored-session-markdown
```

Enter the separately stored passphrase when GnuPG prompts for it.
