# Docker Setup

This project includes Docker support for easy deployment and development.

## Prerequisites

- Docker
- Docker Compose

## Configuration

The application uses port **10020** by default. You can change this in `docker-compose.yml` if needed.

## Environment Variables

Create a `backend/secrets.env` file with the following variables:

```env
OPENROUTER_API_KEY=your_api_key_here
DEBUG=false
```

Or set them directly in `docker-compose.yml` under the `environment` section.

## Usage

### Build and Start

```bash
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f backend
```

### Stop

```bash
docker-compose down
```

### Rebuild After Code Changes

```bash
docker-compose up -d --build
```

### Access the Application

Once running, access the application at:
- Frontend: http://localhost:10020
- API: http://localhost:10020/api/v1/health

## Data Persistence

The `backend/data` directory is mounted as a volume, so all database files and data will persist between container restarts.

## Health Check

The container includes a health check that verifies the API is responding. You can check the health status with:

```bash
docker-compose ps
```

