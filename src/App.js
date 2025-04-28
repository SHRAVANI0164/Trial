import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { auth } from './firebase';
import { onAuthStateChanged } from 'firebase/auth';

import Landing from './components/Landing';       // ← import Landing
import Header from './components/Header';
import Login from './components/Login';
import Signup from './components/Signup';
import Dashboard from './components/Dashboard';
import Profile from './components/Profile';

export default function App() {
  const [user, setUser] = useState(null);
  useEffect(() => onAuthStateChanged(auth, setUser), []);

  return (
    <BrowserRouter>
      {/* We’ll still show Navbar on all routes if you like: */}
      <Header user={user} />

      <Routes>
        <Route path="/" element={<Landing />} />

        <Route path="/login"  element={!user ? <Login/>    : <Navigate to="/dashboard"/>}/>
        <Route path="/signup" element={!user ? <Signup/>   : <Navigate to="/dashboard"/>}/>
        <Route path="/dashboard" element={user ? <Dashboard/> : <Navigate to="/login"/>}/>
        <Route path="/landing" element={user ? <Landing/> : <Navigate to="/dashboard"/>}/>
        <Route path="/profile"   element={user ? <Profile/>   : <Navigate to="/login"/>}/>

        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </BrowserRouter>
  );
}
