import { initializeApp } from 'firebase/app';
import { getAuth }         from 'firebase/auth';

const config = {
  apiKey: "AIzaSyCx-5A3i1SmcvWXpQd5OomiyxOpSLGVkvo",
  authDomain: "myreactdashboard-3d785.firebaseapp.com",
  projectId: "myreactdashboard-3d785",
  storageBucket: "myreactdashboard-3d785.firebasestorage.app",
  messagingSenderId: "60051759564",
  appId: "1:60051759564:web:2fb643768926b106d1fd69"
};

const app = initializeApp(config);
export const auth = getAuth(app);
