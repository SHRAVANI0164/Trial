import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { signOut } from 'firebase/auth';
import { auth } from '../firebase';


export default function Navbar({ user }) {
  const loc = useLocation();
  const isLanding = loc.pathname === '/';

  return (
    <nav className="navbar container">
      <Link to="/" className="logo">Skywriter</Link>
      <div className="nav-links">
        {user ? (
          <>
            <Link to="/dashboard">Dashboard</Link>
            <Link to="/profile">Profile</Link>
            <button onClick={()=>signOut(auth)} className="button outline small">Sign Out</button>
          </>
        ) : isLanding ? null /* landing shows its own buttons */ : (
          <>
            <Link to="/login">Login</Link>
            <Link to="/signup" className="button small">Sign Up</Link>
          </>
        )}
      </div>
    </nav>
  );
}
