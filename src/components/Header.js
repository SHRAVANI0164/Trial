import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { signOut } from 'firebase/auth';
import { auth } from '../firebase';

export default function Header({ user }) {
  const loc = useLocation();
  const onLanding = loc.pathname === '/';

  return (
    <header className="header container flex space-between">
      <h1 className="logo">Skywriter</h1>

      <div className="header-links">
        {user ? (
          <>
            <Link to="/dashboard">Dashboard</Link>
            <Link to="/profile">Profile</Link>
            <button className="button outline small" onClick={() => signOut(auth)}>
              Sign Out
            </button>
          </>
        ) : onLanding ? (
          <>
            <Link to="/login" className="button outline small">Login</Link>
            <Link to="/signup" className="button small">Sign Up</Link>
            <Link to="/signup" className="button cta">Get Started</Link>
          </>
        ) : (
          <>
            <Link to="/login">Login</Link>
            <Link to="/signup" className="button small">Sign Up</Link>
          </>
        )}
      </div>
    </header>
  );
}
