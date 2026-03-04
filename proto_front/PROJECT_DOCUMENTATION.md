# CogniFlow - AI Academic Co-Pilot

## Project Overview

CogniFlow is a high-fidelity desktop sidebar application designed as an AI Academic Co-Pilot. It features a professional glassmorphism dark mode UI that provides real-time cognitive analytics, video monitoring, and AI-powered interventions to help users maintain focus and productivity during study or work sessions.

---

## File Structure

```
/
├── src/
│   ├── app/
│   │   ├── App.tsx                          # Main application entry point
│   │   └── components/
│   │       ├── CogniFlowSidebar.tsx         # Main sidebar component with all features
│   │       ├── figma/
│   │       │   └── ImageWithFallback.tsx    # Image component with fallback (protected)
│   │       └── ui/                          # Radix UI component library
│   │           ├── button.tsx
│   │           ├── card.tsx
│   │           ├── dialog.tsx
│   │           └── ... (50+ UI components)
│   │
│   └── styles/
│       ├── fonts.css                        # Font imports (Inter font)
│       ├── index.css                        # Global styles
│       ├── tailwind.css                     # Tailwind directives
│       └── theme.css                        # Theme tokens and variables
│
├── package.json                             # Dependencies and scripts
├── vite.config.ts                           # Vite configuration
└── postcss.config.mjs                       # PostCSS configuration
```

---

## Technologies & Libraries Used

### Core Framework

- **React 18.3.1** - UI library
- **TypeScript** - Type safety
- **Vite 6.3.5** - Build tool and dev server

### Styling

- **Tailwind CSS v4.1.12** - Utility-first CSS framework
- **Tailwind Merge** - Merge Tailwind classes efficiently
- **Class Variance Authority** - Component variant management
- **Inter Font** - Primary sans-serif font (Google Fonts)

### UI Component Library

- **Radix UI** - Unstyled, accessible component primitives
  - Dialog, Popover, Tooltip, Accordion, etc.
- **Lucide React 0.487.0** - Icon library (Settings, ChevronRight, etc.)

### Animation

- **Motion (Framer Motion) 12.23.24** - Animation library (not yet used, but available)

### Available Libraries (Not Currently Used)

- **Recharts** - For potential data visualization
- **React Router 7.13.0** - For multi-page navigation
- **React Hook Form** - For form management
- **Material UI** - Alternative component library
- **Sonner** - Toast notifications

---

## Current Features

### 1. **Collapsible Sidebar**

- **Collapsed State**: 20px trigger strip with vertical "CogniFlow" label
- **Expanded State**: 320px full sidebar
- Smooth transitions (500ms ease-in-out)

### 2. **Header (Status Bar)**

- Pulsing "Flow State" indicator (neon green)
- Live digital clock (HH:MM:SS format)
- Collapse/expand button

### 3. **Video Feed Module**

- 280x210px camera viewport
- Face bounding box (140x180px) with neon green border
- "USER: FLOW STATE" label above bounding box
- White eye-tracking dot on user's eye
- Animated gaze target overlay (moving crosshair)
- Scanline overlay effect
- Recording indicator (pulsing red dot + "REC" label)

### 4. **Cognitive Analytics Section**

- **Focus Score**: Progress bar (0-100%) with color transitions
  - Green (#00ff00) for 70%+
  - Orange (#ffa500) for 40-69%
  - Red (#e74c3c) for <40%
- **Recovery Latency (FRT)**: Tag showing response time in seconds

### 5. **AI Intervention Area**

- Conversational chat bubbles from Gemini AI
- Message timestamps
- "AI is thinking" animation (three pulsing dots)
- Different bubble styles for different message types

### 6. **Footer**

- "STOP SESSION" button (muted red #e74c3c)
- Settings icon button

### 7. **Glassmorphism Design**

- Background: rgba(18, 18, 18, 0.6)
- 20px backdrop blur
- Rounded left corners (20px), flush right edge
- Neon green accents (#00ff00)
- Soft red for warnings/stops (#e74c3c)

### 8. **Custom Scrollbar**

- Styled scrollbar for entire sidebar
- Neon green thumb matching theme

---

## Backend Integration Requirements

To make CogniFlow fully functional, you'll need to implement the following backend services:

### 1. **Real-Time Video Processing**

#### Technologies Needed:

- **WebRTC** or **MediaStream API** - Access user's webcam
- **TensorFlow.js** or **MediaPipe** - Face detection and eye tracking
- **OpenCV** (Python backend) - Advanced computer vision

#### Implementation Steps:

```javascript
// Frontend: Access webcam
const stream = await navigator.mediaDevices.getUserMedia({
  video: { width: 280, height: 210 }
});

// Send video frames to backend for processing
// Receive face bounding box coordinates and gaze position
```

#### Backend API Endpoints:

```
POST /api/video/analyze
- Input: Video frame (base64 or blob)
- Output: {
    faceBounds: { x, y, width, height },
    eyePosition: { x, y },
    gazeTarget: { x, y }
  }
```

---

### 2. **Cognitive State Detection**

#### Technologies Needed:

- **AI/ML Model** - Analyze facial expressions, posture, eye movement
- **TensorFlow** or **PyTorch** - Model training and inference
- **Real-time data streaming** - WebSocket or Server-Sent Events

#### Metrics to Track:

- **Focus Score** (0-100%)
- **Flow State** ("Flow State", "Thinking", "Distracted", "Fatigued")
- **Recovery Latency** (time to refocus after distraction)

#### Backend API Endpoints:

```
POST /api/cognitive/analyze
- Input: {
    gazeStability: number,
    blinkRate: number,
    facialExpression: string,
    headPose: { pitch, yaw, roll }
  }
- Output: {
    focusScore: number,
    currentState: string,
    recoveryLatency: number
  }

WebSocket: ws://localhost:8080/cognitive-stream
- Real-time updates for focus metrics
```

---

### 3. **AI Intervention System (Gemini Integration)**

#### Technologies Needed:

- **Google Gemini API** - AI responses
- **Prompt Engineering** - Context-aware interventions
- **Session History** - Track user patterns

#### Implementation:

```javascript
// Frontend: Send session data to AI
const response = await fetch('/api/ai/intervention', {
  method: 'POST',
  body: JSON.stringify({
    focusScore: 82,
    sessionDuration: 32,
    currentState: 'Flow State',
    recentEvents: ['distraction at 15min', 'recovered in 1.2s']
  })
});
```

#### Backend API Endpoints:

```
POST /api/ai/intervention
- Input: {
    focusScore: number,
    sessionDuration: number,
    currentState: string,
    userContext: object
  }
- Output: {
    message: string,
    interventionType: 'encouragement' | 'warning' | 'suggestion',
    timestamp: string
  }

GET /api/ai/chat-history
- Returns: Array of previous AI messages
```

---

### 4. **Session Management**

#### Technologies Needed:

- **Database** - PostgreSQL, MongoDB, or Firebase
- **Authentication** - JWT or OAuth
- **Session Storage** - Redis for active sessions

#### Features to Implement:

- Start/stop session tracking
- Save session analytics
- Generate productivity reports
- User preferences storage

#### Backend API Endpoints:

```
POST /api/session/start
- Input: { userId: string }
- Output: { sessionId: string, startTime: timestamp }

POST /api/session/stop
- Input: { sessionId: string }
- Output: {
    sessionId: string,
    duration: number,
    averageFocus: number,
    interventionCount: number
  }

GET /api/session/history
- Returns: Array of past sessions with analytics
```

---

### 5. **Database Schema**

```sql
-- Users table
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email VARCHAR(255) UNIQUE,
  name VARCHAR(255),
  created_at TIMESTAMP DEFAULT NOW()
);

-- Sessions table
CREATE TABLE sessions (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  start_time TIMESTAMP,
  end_time TIMESTAMP,
  average_focus_score FLOAT,
  peak_focus_score FLOAT,
  total_distractions INT,
  average_recovery_latency FLOAT,
  session_notes TEXT
);

-- Metrics table (time-series data)
CREATE TABLE session_metrics (
  id UUID PRIMARY KEY,
  session_id UUID REFERENCES sessions(id),
  timestamp TIMESTAMP,
  focus_score FLOAT,
  gaze_x FLOAT,
  gaze_y FLOAT,
  current_state VARCHAR(50),
  blink_rate INT
);

-- AI Interventions table
CREATE TABLE ai_interventions (
  id UUID PRIMARY KEY,
  session_id UUID REFERENCES sessions(id),
  timestamp TIMESTAMP,
  message TEXT,
  intervention_type VARCHAR(50),
  user_response VARCHAR(50)
);
```

---

### 6. **Recommended Backend Stack**

#### Option 1: Node.js + Express

```bash
npm install express ws socket.io @google-ai/generativelanguage
npm install pg redis jsonwebtoken bcrypt
```

#### Option 2: Python + FastAPI

```bash
pip install fastapi uvicorn websockets
pip install opencv-python mediapipe tensorflow
pip install google-generativeai sqlalchemy redis
```

#### Option 3: Supabase (Recommended for Fast Setup)

- Built-in PostgreSQL database
- Real-time subscriptions
- Authentication out-of-the-box
- Edge Functions for serverless backend
- Storage for video recordings

---

### 7. **Real-Time Communication Architecture**

```
Frontend (React)
    ↓ WebSocket
WebSocket Server
    ↓
┌─────────────────┬──────────────────┬───────────────┐
│  Video Processor │  Cognitive AI    │  Gemini API   │
│  (Face Detection)│  (Focus Analysis)│  (Chat Bot)   │
└─────────────────┴──────────────────┴───────────────┘
    ↓                   ↓                   ↓
                  Database (PostgreSQL)
```

---

### 8. **Environment Variables Needed**

```env
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/cogniflow

# Redis (for sessions)
REDIS_URL=redis://localhost:6379

# JWT Secret
JWT_SECRET=your_secret_key_here

# Frontend URL
FRONTEND_URL=http://localhost:5173

# WebSocket Port
WS_PORT=8080
```

---

### 9. **Frontend Integration Points**

Update `CogniFlowSidebar.tsx` to connect to backend:

```typescript
// Replace mock data with API calls
const [focusScore, setFocusScore] = useState(82);
const [recoveryLatency, setRecoveryLatency] = useState(1.2);
const [flowState, setFlowState] = useState("Flow State");

// Connect to WebSocket for real-time updates
useEffect(() => {
  const ws = new WebSocket(
    "ws://localhost:8080/cognitive-stream",
  );

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    setFocusScore(data.focusScore);
    setFlowState(data.currentState);
    setRecoveryLatency(data.recoveryLatency);
  };

  return () => ws.close();
}, []);

// Start webcam and send frames to backend
useEffect(() => {
  const startVideo = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
    });
    videoRef.current.srcObject = stream;

    // Send frames to backend every 500ms
    const interval = setInterval(() => {
      captureFrameAndSend();
    }, 500);

    return () => clearInterval(interval);
  };

  startVideo();
}, []);
```

---

### 10. **Security Considerations**

- **Camera Permissions**: Request explicit user consent
- **Data Privacy**: Encrypt video data in transit (HTTPS/WSS)
- **GDPR Compliance**: Allow users to delete their data
- **Rate Limiting**: Prevent API abuse
- **Authentication**: Secure all API endpoints
- **Local Processing**: Consider processing video locally to reduce privacy concerns

---

## Development Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

---

## Next Steps for Full Implementation

1. **Set up backend infrastructure** (Node.js/Python + database)
2. **Integrate webcam access** with MediaStream API
3. **Implement face detection** using MediaPipe or TensorFlow.js
4. **Connect Gemini API** for AI interventions
5. **Add WebSocket connection** for real-time metrics
6. **Implement session persistence** with database
7. **Add user authentication** and session management
8. **Deploy backend** (AWS, Google Cloud, or Vercel)
9. **Test end-to-end flow** with real users
10. **Optimize performance** and add analytics

---

## Notes

- Currently, all data is **mock data** for UI demonstration
- **No backend is connected** - this is a frontend-only prototype
- Video feed shows placeholder - needs real webcam integration
- Focus scores and gaze tracking are simulated with random values
- AI messages are hardcoded - needs Gemini API integration

---

## Contact & Support

For backend implementation assistance or questions, refer to:

- Google Gemini API Documentation
- MediaPipe Face Detection Guide
- Supabase Real-time Documentation
- TensorFlow.js Face Landmarks Detection