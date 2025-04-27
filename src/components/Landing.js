import React from 'react';
import { Link } from 'react-router-dom';

export default function Landing() {
  return (
    <div className="landing">
      <main className="landing-hero container">
        <h2>Welcome to MyApp</h2>
        <p>Capture, organize and access your notes anywhere.</p>
        <Link to="/signup" className="button cta large">Get Started Free</Link>
      </main>
    </div>
  );
}
