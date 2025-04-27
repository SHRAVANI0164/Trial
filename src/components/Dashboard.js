import React from 'react';
import { auth } from '../firebase';

export default function Dashboard() {
  return (
    <div className="page">
      <h1>Dashboard</h1>
      <p>Welcome, {auth.currentUser?.email}</p>
    </div>
  );
}
