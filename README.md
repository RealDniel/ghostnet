# ghostnet

## Prerequisites

- [Node.js](https://nodejs.org/) v18 or higher (includes npm)
- Backend server running on `http://localhost:8000`

### Installing Node.js

**Windows:** Download the installer from [nodejs.org](https://nodejs.org/) and run it.

**macOS:**
```bash
# Using Homebrew (recommended)
brew install node

# Or download the installer from nodejs.org
```
If you don't have Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

## Frontend Dependencies

| Package | Version | Purpose |
|---|---|---|
| react | ^18.3.1 | UI framework |
| react-dom | ^18.3.1 | React DOM renderer |
| vite | ^5.4.10 | Dev server and bundler |
| @vitejs/plugin-react | ^4.3.1 | React support for Vite |
| tailwindcss | ^3.4.14 | Utility-first CSS |
| postcss | ^8.4.47 | CSS processing |
| autoprefixer | ^10.4.20 | CSS vendor prefixes |

## Backend API Expected

The frontend expects the following from the backend:

- `ws://localhost:8000/ws` — WebSocket stream of live events
- `GET http://localhost:8000/events` — Returns array of historical events

### WebSocket message schema

```json
{ "event": "fall_detected", "confidence": 0.94, "timestamp": "2026-05-30T18:42:00Z" }
{ "event": "presence_update", "occupied": true, "timestamp": "2026-05-30T18:42:00Z" }
```
