# Encrypted session archive

This directory contains an encrypted Markdown export of the machine's AI sessions.
The original session files were not deleted or modified.

- Snapshot date: 2026-08-26 UTC
- Session Markdown files: 31
- Exported messages: 45,613
- Archive encryption: GnuPG symmetric AES-256
- Compression: Zstandard
- SHA-256: `30de526039d2f31c8a94f9f5a864c5671e640fad8daf65a5f5aaf5ec2f807427`

The passphrase is intentionally not stored in this repository. Run:

```bash
./secure-session-backup/restore.sh \
  ./secure-session-backup/sessions-20260826.tar.zst.gpg \
  ./restored-session-markdown
```

Enter the separately stored passphrase when GnuPG prompts for it.
