+++
date = '2026-06-24T00:00:00+05:30'
draft = false
title = 'What Turso Is'
slug = 'what-turso-is'
summary = 'An overview of Turso, libSQL, Limbo, and the SQLite rewrite — what each piece is, why Turso is rewriting SQLite in Rust, and when to use which.'
kicker = 'Database'
toc = true
showreadingtime = true
+++

The naming is a little messy. In current official docs, libSQL is the older SQLite fork, while Turso Database is the new ground-up rewrite of SQLite in Rust. Turso says the rewrite is meant for concurrent writes, async I/O, and local-first sync. The company also says Turso Cloud still runs on libSQL today and will integrate the new engine later.

The rewrite was first introduced publicly in late 2024 as Limbo, then Turso said in January 2025 that it was going all-in on the rewrite and turning it into an official project. The goal stated in the announcement was blunt: reimplement SQLite from scratch, keep language and file-format compatibility, and get the same or better reliability with Rust memory safety and a modern architecture.

## Why Turso is doing this

SQLite is brilliant for local storage. SQLite's own docs say it is built for local data storage, embedded devices, apps, websites, caches, and self-contained files, and it is usually the right choice when writer concurrency is low. But the same docs also say SQLite allows only one writer at a time per database file, and that many concurrent writers may need a different solution.

That concurrency ceiling is the real reason Turso exists. Turso's docs say its default mode still limits writes, but its MVCC mode and BEGIN CONCURRENT let multiple connections write at once, with conflicts detected only when two transactions touch the same rows. In other words, Turso is trying to keep SQLite's simplicity while pushing past its biggest bottleneck for modern multi-writer apps.

## How it works

Turso Database is an in-process SQL database written in Rust, compatible with SQLite. Its own GitHub README lists SQLite compatibility at the SQL, file-format, and C-API levels, plus BEGIN CONCURRENT, change data capture, multi-language support, async I/O on Linux via io_uring, cross-platform support, vector features, schema improvements, encryption, and experimental multi-process WAL coordination.

The technical idea is simple but ambitious: keep the SQLite mental model, but re-architect the engine around modern concurrency and reliability techniques. Turso says the rewrite bakes in Deterministic Simulation Testing from the start and pairs it with Antithesis to stress the engine under hard-to-reproduce execution paths. The rewrite is also meant to stay compatible at the language and file-format level so existing SQL and schema can keep working.

On the product side, Turso now splits the ecosystem into packages. Official docs recommend @tursodatabase/database, pyturso, tursogo, or turso for new local/embedded work on the new engine, @tursodatabase/sync for explicit push/pull cloud sync, and @tursodatabase/serverless for remote Turso Cloud access in serverless and edge environments. Existing @libsql/client users are told that libSQL remains production-ready and battle-tested.

## The main benefits

First, more write concurrency. Turso's concurrent-writes docs say MVCC lets multiple connections write simultaneously, and non-overlapping transactions proceed without conflict. That is the headline feature.

Second, Rust memory safety. Turso explicitly says the rewrite is intended to have full memory safety. That matters in a database engine, where memory bugs become data bugs.

Third, SQLite compatibility with a modern core. Turso says the rewrite is designed to keep SQLite's language and file-format compatibility, so the idea is not to force you into a new SQL dialect or a new schema world.

Fourth, local-first and sync-friendly behavior. The docs describe local databases that can push and pull to Turso Cloud, with reads and writes happening locally in sync mode. That is useful for offline-first apps, devices, edge deployments, and apps that need local speed with cloud coordination.

Fifth, a narrower deployment surface. The serverless package uses only fetch, with zero native dependencies, which makes it easier to run in Node, Docker, serverless, and edge runtimes. That is a practical advantage, not a marketing one.

## The tradeoffs

The biggest one is maturity. The Turso Database repo still warns that the software is beta and may contain bugs and unexpected behavior. Turso's own docs therefore recommend libSQL for "mission-critical workloads that need a battle-tested foundation today," while recommending the rewrite for new projects.

The second tradeoff is conflict handling. With MVCC, concurrent writers are possible, but your app must detect conflict errors and retry transactions. That is better than a hard single-writer bottleneck, but it is not magic; it shifts some complexity into application logic.

The third is compatibility boundaries. Turso says encrypted databases cannot be read as standard SQLite databases; they must be opened with the Turso Database engine. So while the goal is compatibility, some features deliberately break plain SQLite interoperability.

## When to use it

Use plain SQLite when you want the simplest answer: local storage, low write concurrency, single-file portability, and a long-proven engine. SQLite itself still recommends it for embedded devices, small apps, caches, teaching, demos, and many low-to-medium traffic websites.

Use libSQL / Turso Cloud today when you need a production-ready path with SQLite compatibility and managed cloud features, especially if you already rely on @libsql/client or need mature ORM support. Turso's docs call this the battle-tested choice for mission-critical workloads.

Use Turso Database when your new project needs the rewrite's core promises: concurrent writes, local-first sync, Rust safety, and a modern architecture without giving up SQLite-like semantics. That is the direction Turso is pushing for new builds.

Turso is trying to do something narrow and hard: preserve what people love about SQLite, then remove the parts that hurt in modern apps. The rewrite is real, but it is still early. If you need the most stable choice today, libSQL is the safer Turso path. If you are building new and can accept beta software, Turso Database is the interesting bet.
