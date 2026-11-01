import React from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import 'regenerator-runtime/runtime';
import App from './Studio'




ReactDOM.createRoot(document.getElementById('root')).render(

    <HashRouter>
      <App />
    </HashRouter>

)
