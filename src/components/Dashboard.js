import React from 'react';
import { auth } from '../firebase';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  return (
    <div className="page">
      <h1>Dashboard</h1>
      <p>Welcome, {auth.currentUser?.email}</p>
      <Link to="/landing"><button value="Get started">Get Started</button></Link>
    </div>
  );
}
