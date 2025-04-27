import React, {useState} from 'react';
import { signInWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../firebase';

export default function Login() {
  const [email,setEmail]=useState(''), [pw,setPw]=useState(''), [err,setErr]=useState('');
  const submit=e=>{
    e.preventDefault();
    signInWithEmailAndPassword(auth,email,pw)
      .catch(e=>setErr(e.message));
  };
  return (
    <form onSubmit={submit} className="auth-form">
      <h2>Login</h2>
      {err && <p className="error">{err}</p>}
      <input type="email"    value={email} onChange={e=>setEmail(e.target.value)} placeholder="Email" required/>
      <input type="password" value={pw}    onChange={e=>setPw(e.target.value)} placeholder="Password" required/>
      <button type="submit">Login</button>
    </form>
  );
}
