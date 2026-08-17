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

  // Google Signals fans measurement out to ad domains: stats.g.doubleclick.net,
  // www.google.com, and www.google.<cctld>/ads/ga-audiences. That last one
  // varies by visitor country, so it cannot be expressed in a CSP allowlist
  // without enumerating every ccTLD. This site runs no ads and no remarketing,
  // so the data was unused. Turning the signals off keeps page views intact,
  // keeps connect-src tight, and avoids sending visitors to ad domains at all.
  gtag('config', GA_ID, {
    allow_google_signals: false,
    allow_ad_personalization_signals: false,
  })
  
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