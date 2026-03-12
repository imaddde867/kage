# Kage Product Scope

Kage is a local personal assistant for one user on one macOS machine.

## Core identity

- Fully local inference by default
- Voice-first interaction, with text chat as a fast local fallback
- Private-by-default memory stored in local SQLite
- Small enough to understand, operate, and debug without platform teams

## Explicit non-goals

- Multi-channel messaging integrations such as WhatsApp, Telegram, Slack, or Discord
- Remote gateway or control-plane architecture
- Multi-user or team session routing
- Browser automation platform
- Cross-platform product surface across iOS, Android, web, and desktop nodes

## What to borrow from larger assistant platforms

- Better onboarding and diagnostics
- Better backup and restore ergonomics
- Better docs around permissions and safety
- Better packaging polish for daily use

## What not to copy

- Channel breadth for its own sake
- Remote-surface security burden
- Host-wide capability expansion that weakens the local-first safety model
