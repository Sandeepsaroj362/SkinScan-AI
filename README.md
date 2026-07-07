# SkinScan AI
Upgraded version is skinscan and Go to that repo.
AI-powered skin disease detection platform built using Deep Learning, FastAPI, Next.js, and Explainable AI.
<img width="1440" height="900" alt="Screenshot 2026-05-07 at 11 01 25 PM" src="https://github.com/user-attachments/assets/36bdb8eb-6894-4254-8192-2633bdca14dd" />
<img width="1440" height="900" alt="Screenshot 2026-05-07 at 11 01 44 PM" src="https://github.com/user-attachments/assets/525d0815-e1be-4c56-bb9b-e9618d1f5d18" />
<img width="1440" height="900" alt="Screenshot 2026-05-07 at 11 02 36 PM" src="https://github.com/user-attachments/assets/498b9fca-3168-4730-93b1-ec09bfa7a171" />
<img width="1440" height="900" alt="Screenshot 2026-05-07 at 11 02 52 PM" src="https://github.com/user-attachments/assets/b19aae38-a5fb-4ccb-8343-9f2fcd99060b" />
<img width="1440" height="900" alt="Screenshot 2026-05-07 at 11 03 14 PM" src="https://github.com/user-attachments/assets/52ba0339-a9b1-4211-b814-63f28a1f8c2b" />
<img width="1440" height="900" alt="Screenshot 2026-05-07 at 11 07 37 PM" src="https://github.com/user-attachments/assets/f25b6400-7988-454a-a5d4-8cc77b680e64" />
<img width="1440" height="900" alt="Screenshot 2026-05-07 at 11 08 06 PM" src="https://github.com/user-attachments/assets/cc6173c3-efaa-4a8b-ae4e-dc8b79d23fe6" />



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
