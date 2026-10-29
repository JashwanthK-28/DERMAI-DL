import React, { useState } from 'react';
import { authApi } from '../api/client';
import { Button, Input, Card, Badge } from '../components/ui';

export default function Login({ onLoginSuccess }) {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('clinician');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (isRegister) {
        await authApi.register(email, password, role);
      }
      const data = await authApi.login(email, password);
      localStorage.setItem('token', data.access_token);
      localStorage.setItem(
        'user',
        JSON.stringify({ id: data.user_id, email: data.email, role: data.role })
      );
      if (onLoginSuccess) {
        onLoginSuccess(data);
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  const title = isRegister ? 'Create your account' : 'Welcome back';
  const subtitle = isRegister
    ? 'Join the DermAI clinical platform'
    : 'Sign in to access skin lesion analysis';

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-slate-50">
      <div className="w-full max-w-md">
        <Card className="shadow-xl border-slate-100">
          <div className="px-6 pt-6 pb-4">
            <div className="flex items-center gap-3 justify-center mb-3">
              <div className="w-12 h-12 bg-gradient-to-br from-indigo-600 to-indigo-400 rounded-2xl flex items-center justify-center text-white text-2xl font-bold shadow-lg shadow-indigo-200">
                D
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900 tracking-tight">DermAI</h1>
                <p className="text-xs text-slate-500">
                  Clinical Skin Lesion Analysis Platform
                </p>
              </div>
            </div>

            <h2 className="text-base font-bold text-slate-900 text-center">
              {title}
            </h2>
            <p className="text-xs text-slate-500 text-center mt-1">
              {subtitle}
            </p>
          </div>

          {error && (
            <div className="px-6 pb-3">
              <div className="bg-rose-50 border border-rose-200 text-rose-700 text-xs p-3 rounded-xl">
                {error}
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="px-6 pb-6 space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-600 tracking-tight uppercase mb-1">
                Email Address
              </label>
              <input
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="clinician@hospital.org"
                className="w-full px-3.5 py-2.5 bg-white border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-600 tracking-tight uppercase mb-1">
                Password
              </label>
              <input
                type="password"
                required
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-3.5 py-2.5 bg-white border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition"
              />
            </div>

            {isRegister && (
              <div>
                <label className="block text-xs font-semibold text-slate-600 tracking-tight uppercase mb-1">
                  User Role
                </label>
                <select
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  className="w-full px-3.5 py-2.5 bg-white border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition"
                >
                  <option value="clinician">Clinician (Dermatologist)</option>
                  <option value="researcher">Researcher</option>
                  <option value="patient">Patient</option>
                  <option value="admin">Administrator</option>
                </select>
              </div>
            )}

            <Button
              type="submit"
              variant="primary"
              size="lg"
              disabled={loading}
              className="w-full shadow-sm shadow-indigo-100"
            >
              {loading ? 'Authenticating...' : isRegister ? 'Create Account' : 'Sign In'}
            </Button>
          </form>

          <div className="px-6 pb-6">
            <div className="relative flex items-center center my-3">
              <div className="flex-1 border-t border-slate-200"></div>
              <span className="px-2 text-[10px] text-slate-400 uppercase font-semibold">
                Or continue with
              </span>
              <div className="flex-1 border-t border-slate-200"></div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="ghost"
                size="sm"
                disabled={loading}
                onClick={() => {
                  setError(null);
                }}
              >
                Google
              </Button>
              <Button
                variant="ghost"
                size="sm"
                disabled={loading}
                onClick={() => {
                  setError(null);
                }}
              >
                Microsoft
              </Button>
            </div>
          </div>

          <div className="px-6 pt-3 text-center border-t border-slate-100">
            <button
              onClick={() => {
                setIsRegister(!isRegister);
                setError(null);
              }}
              className="text-xs text-indigo-600 font-semibold hover:text-indigo-700 transition underline-offset-4 hover:underline"
            >
              {isRegister
                ? 'Already have an account? Sign in'
                : "Don't have an account? Register"}
            </button>
          </div>
        </Card>

        <div className="mt-6 text-center">
          <Badge variant="neutral" className="text-[10px]">
            Infosys Springboard AI Capstone
          </Badge>
        </div>
      </div>
    </div>
  );
}
