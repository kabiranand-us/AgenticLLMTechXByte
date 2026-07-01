# Judge0 for Local Development

This directory contains a `docker-compose.yml` file to run Judge0 locally.

## Usage

To start the Judge0 stack, run:

```bash
docker-compose up -d
```

This will start the following services:

- Judge0 API
- Judge0 Workers
- PostgreSQL
- Redis

The Judge0 API will be available at `http://localhost:2358`.
