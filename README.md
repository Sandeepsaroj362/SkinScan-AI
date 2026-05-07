# Skin AI

AI-powered skin disease detection platform built using Deep Learning, FastAPI, Next.js, and Explainable AI.

Skin AI allows users to upload skin lesion images and receive:
- AI disease prediction
- Confidence score
- Grad-CAM explainability heatmaps
- AI-generated medical explanations
- Downloadable PDF reports
- Consultation booking workflow
- Analytics dashboard

---

# Features

## AI Disease Detection
- EfficientNet-based image classification model
- Trained on skin lesion datasets
- High-confidence prediction pipeline

## Explainable AI
- Grad-CAM heatmaps
- Visual attention mapping
- Transparent AI prediction reasoning

## AI Medical Explanations
- LLM-generated disease explanations
- Symptoms and precautions
- Consultation recommendations

## Authentication System
- Secure login/signup
- Session-based authentication
- Protected routes

## Consultation Workflow
- Free online consultation booking
- Physical dermatologist appointment booking
- Razorpay payment integration

## Analytics Dashboard
- Disease distribution charts
- Scan activity analytics
- High-risk case monitoring

## PDF Report Generation
- Downloadable AI medical reports
- Prediction summaries
- Patient-friendly reports

---

# Tech Stack

## Frontend
- Next.js
- TypeScript
- Tailwind CSS
- Recharts
- NextAuth

## Backend
- FastAPI
- TensorFlow
- Python

## Machine Learning
- EfficientNetB0
- Grad-CAM Explainability
- TensorFlow/Keras

## Database
- MongoDB Atlas

## Cloud Services
- Cloudinary
- Razorpay

---

# Project Architecture

Frontend (Next.js)
↓
FastAPI Backend
↓
TensorFlow Model
↓
MongoDB Atlas
↓
Cloudinary Storage
↓
AI Explanation Engine

---

# Screenshots

## Homepage
Modern healthcare SaaS UI with AI consultation workflow.

## AI Prediction
- Disease prediction
- Confidence score
- Grad-CAM visualization

## Dashboard
- Analytics
- Charts
- Risk monitoring

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/skin-ai.git
cd skin-ai

```
#Frontend Setup
```
cd frontend

npm install

npm run dev
```
#Backend Setup
```
cd backend

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8001
```

#Frontend .env.local
```
NEXT_PUBLIC_API_URL=http://localhost:8001

NEXTAUTH_SECRET=your_secret

NEXTAUTH_URL=http://localhost:3000

MONGODB_URI=your_mongodb_uri

NEXT_PUBLIC_RAZORPAY_KEY=your_razorpay_key
```
#Backend .env
```

MONGODB_URI=your_mongodb_uri

CLOUDINARY_CLOUD_NAME=your_cloud_name

CLOUDINARY_API_KEY=your_api_key

CLOUDINARY_API_SECRET=your_api_secret

GEMINI_API_KEY=your_api_key

RAZORPAY_KEY_ID=your_key

RAZORPAY_KEY_SECRET=your_secret
```

## Machine Learning Pipeline
Image Upload
Image Preprocessing
EfficientNet Prediction
Confidence Calculation
Grad-CAM Heatmap Generation
AI Medical Explanation
Store Prediction History
Generate Reports

##Author
Atul Pal
GitHub: https://github.com/atulpal02
LinkedIn: https://linkedin.com/in/atulpal02

##License
This project is built for educational and research purposes.
