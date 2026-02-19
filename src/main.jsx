import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

// Initialize Google Analytics
const GA_ID = import.meta.env.VITE_GA_ID

if (GA_ID) {
  // Initialize dataLayer first (acts as a queue)
  window.dataLayer = window.dataLayer || []
  function gtag() { window.dataLayer.push(arguments) }
  window.gtag = gtag
  
  // Record timestamp first
  gtag('js', new Date())
  gtag('config', GA_ID)
  
  // Load script next (async, so dataLayer queue handles earlier calls)
  const script = document.createElement('script')
  script.async = true
  script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_ID}`
  document.head.appendChild(script)
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)