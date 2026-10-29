import React, { useState, useEffect } from 'react';
import { casesApi } from '../api/client';
import {
  Button,
  Input,
  Card,
  Badge,
  Drawer,
  Section,
  Skeleton,
} from '../components/ui';

const SITE_LABELS = {
  FOREARM: 'Forearm',
  FACE: 'Face',
  CHEST: 'Chest',
  BACK: 'Back',
  NECK: 'Neck',
  NOSE: 'Nose',
  ARM: 'Arm',
  HAND: 'Hand',
  SCALP: 'Scalp',
  EAR: 'Ear',
  THIGH: 'Thigh',
};

const SEX_LABELS = { FEMALE: 'Female', MALE: 'Male', OTHER: 'Other' };

function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

function countToday(items) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return items.filter((c) => {
    const d = new Date(c.created_at);
    d.setHours(0, 0, 0, 0);
    return d.getTime() === today.getTime();
  }).length;
}

function countThisWeek(items) {
  const now = new Date();
  const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
  return items.filter((c) => new Date(c.created_at) >= weekAgo).length;
}

export default function Dashboard({ user, onSelectCase, onLogout }) {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [openCreate, setOpenCreate] = useState(false);
  const [formData, setFormData] = useState({
    patient_clinical_id: '',
    age: '',
    sex: 'FEMALE',
    anatomic_site: 'FOREARM',
    notes: '',
  });
  const [formError, setFormError] = useState(null);
  const [creating, setCreating] = useState(false);

  const loadCases = async () => {
    try {
      setLoading(true);
      const data = await casesApi.list();
      setCases(data);
    } catch (err) {
      console.error('Failed to load cases:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCases();
  }, []);

  const handleCreateCase = async (e) => {
    e.preventDefault();
    setFormError(null);
    setCreating(true);
    try {
      const payload = {
        ...formData,
        age: formData.age ? parseInt(formData.age, 10) : null,
      };
      await casesApi.create(payload);
      setOpenCreate(false);
      setFormData({
        patient_clinical_id: '',
        age: '',
        sex: 'FEMALE',
        anatomic_site: 'FOREARM',
        notes: '',
      });
      loadCases();
    } catch (err) {
      setFormError(err.response?.data?.detail || 'Failed to create case');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-gradient-to-br from-indigo-600 to-indigo-400 rounded-xl flex items-center justify-center text-white text-lg font-bold shadow-lg shadow-indigo-200">
              D
            </div>
            <div>
              <h1 className="text-base font-bold text-slate-900 leading-tight">
                DermAI Clinical Dashboard
              </h1>
              <p className="text-[11px] text-slate-400">
                Skin Lesion Analysis System
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex flex-col items-end text-right">
              <p className="text-xs font-semibold text-slate-800">{user?.email}</p>
              <Badge variant="brand" className="text-[10px]">
                {user?.role}
              </Badge>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={onLogout}
              className="text-slate-600 hover:bg-slate-100"
            >
              Sign Out
            </Button>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full space-y-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <Section
            title="Case Overview"
            subtitle="Manage patient cases and trigger deep learning lesion classification."
          />
          <div className="flex items-center gap-3">
            <Button
              variant="primary"
              size="sm"
              onClick={() => setOpenCreate(true)}
              className="shadow-sm shadow-indigo-100"
            >
              + New Case
            </Button>
            <Button variant="outline" size="sm">
              Export CSV
            </Button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card padding="p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">
                  Total Cases
                </p>
                <p className="text-2xl font-black text-slate-900">{cases.length}</p>
              </div>
              <div className="w-10 h-10 bg-indigo-50 text-indigo-600 rounded-xl flex items-center justify-center text-xl">
                📋
              </div>
            </div>
            <p className="text-[10px] text-slate-400 mt-3">
              {countThisWeek(cases)} added this week
            </p>
          </Card>

          <Card padding="p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">
                  Cases Today
                </p>
                <p className="text-2xl font-black text-slate-900">
                  {countToday(cases)}
                </p>
              </div>
              <div className="w-10 h-10 bg-emerald-50 text-emerald-600 rounded-xl flex items-center justify-center text-xl">
                📥
              </div>
            </div>
            <p className="text-[10px] text-slate-400 mt-3">
              Uploaded and queued for analysis
            </p>
          </Card>

          <Card padding="p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">
                  Models Deployed
                </p>
                <p className="text-2xl font-black text-slate-900">2</p>
              </div>
              <div className="w-10 h-10 bg-amber-50 text-amber-600 rounded-xl flex items-center justify-center text-xl">
                🧠
              </div>
            </div>
            <p className="text-[10px] text-slate-400 mt-3">
              ResNet50 + EfficientNet-B0
            </p>
          </Card>

          <Card padding="p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">
                  System Status
                </p>
                <p className="text-sm font-bold text-emerald-700">Online</p>
              </div>
              <div className="w-10 h-10 bg-emerald-100 text-emerald-600 rounded-xl flex items-center justify-center">
                <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse-soft" />
              </div>
            </div>
            <p className="text-[10px] text-slate-400 mt-3">
              API healthy • Last synced just now
            </p>
          </Card>
        </div>

        {/* Cases list */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Your Clinical Cases</h3>
              <p className="text-[10px] text-slate-400">
                Tap a case to review images and AI predictions
              </p>
            </div>
            <span className="text-[10px] text-slate-400 font-medium">
              {cases.length} case{cases.length !== 1 ? 's' : ''}
            </span>
          </div>

          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <Card key={i} padding="p-5">
                  <Skeleton width="40%" height="10px" className="mb-3" />
                  <Skeleton width="100%" height="16px" className="mb-2" />
                  <Skeleton width="60%" height="12px" />
                  <Skeleton width="80%" height="12px" />
                </Card>
              ))}
            </div>
          ) : cases.length === 0 ? (
            <Card className="text-center py-16 border-dashed border-slate-300 hover:border-indigo-400 transition">
              <div className="w-14 h-14 bg-slate-100 text-slate-400 rounded-2xl flex items-center justify-center mx-auto text-2xl mb-3">
                📋
              </div>
              <h4 className="text-base font-semibold text-slate-800">No Cases Yet</h4>
              <p className="text-xs text-slate-500 max-w-xs mx-auto mt-2">
                Create your first patient case to begin capturing mobile images and
                triggering the deep learning classification pipeline.
              </p>
              <Button
                variant="primary"
                size="sm"
                onClick={() => setOpenCreate(true)}
                className="mt-4 shadow-sm shadow-indigo-100"
              >
                Create Your First Case
              </Button>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {cases.map((c) => (
                <Card
                  key={c.id}
                  onClick={() => onSelectCase(c)}
                  className="p-5 cursor-pointer hover:shadow-md transition border-slate-200 hover:border-indigo-200 group"
                >
                  <div className="flex items-start justify-between mb-3">
                    <Badge variant="brand" className="text-[10px]">
                      {c.patient_clinical_id || 'Unnamed'}
                    </Badge>
                    <span className="text-[10px] text-slate-400">
                      {formatDate(c.created_at)}
                    </span>
                  </div>
                  <h4 className="text-sm font-bold text-slate-900 mb-2 leading-tight">
                    Site: {SITE_LABELS[c.anatomic_site] || c.anatomic_site}
                  </h4>
                  <div className="grid grid-cols-2 gap-2 text-xs text-slate-500 mb-3">
                    <span className="flex items-center gap-1">
                      <span className="w-1 h-1 rounded-full bg-slate-300" />
                      Age: {c.age ?? 'N/A'}
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-1 h-1 rounded-full bg-slate-300" />
                      {SEX_LABELS[c.sex] || c.sex || 'N/A'}
                    </span>
                  </div>
                  {c.notes && (
                    <p className="text-xs text-slate-500 line-clamp-2 mb-3">
                      {c.notes}
                    </p>
                  )}
                  <div className="flex items-center gap-2 text-xs font-semibold text-indigo-600 bg-indigo-50 px-3 py-2 rounded-xl border border-indigo-100">
                    <span>Review images & AI result</span>
                    <span className="text-indigo-500">→</span>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </main>

      {/* Create case drawer */}
      <Drawer
        isOpen={openCreate}
        onClose={() => setOpenCreate(false)}
        title="Create New Patient Case"
      >
        <form onSubmit={handleCreateCase} className="space-y-4">
          {formError && (
            <div className="bg-rose-50 border border-rose-200 text-rose-700 text-xs p-3 rounded-xl">
              {formError}
            </div>
          )}
          <div>
            <label className="block text-xs font-semibold text-slate-600 tracking-tight uppercase mb-1">
              Clinical Patient ID
            </label>
            <Input
              id="patient-clinical-id"
              type="text"
              required
              placeholder="e.g. PAT_046"
              value={formData.patient_clinical_id}
              onChange={(e) =>
                setFormData({ ...formData, patient_clinical_id: e.target.value })
              }
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-600 tracking-tight uppercase mb-1">
                Age
              </label>
              <Input
                id="age"
                type="number"
                min="1"
                max="120"
                placeholder="e.g. 55"
                value={formData.age}
                onChange={(e) => setFormData({ ...formData, age: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 tracking-tight uppercase mb-1">
                Sex
              </label>
                <select
                  value={formData.sex}
                  onChange={(e) =>
                    setFormData({ ...formData, sex: e.target.value })
                  }
                  className="w-full px-3.5 py-2.5 bg-white border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition"
                >
                  <option value="FEMALE">Female</option>
                  <option value="MALE">Male</option>
                  <option value="OTHER">Other</option>
                </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-600 tracking-tight uppercase mb-1">
              Anatomic Site
            </label>
            <select
              value={formData.anatomic_site}
              onChange={(e) =>
                setFormData({ ...formData, anatomic_site: e.target.value })
              }
              className="w-full px-3.5 py-2.5 bg-white border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition"
            >
              {Object.entries(SITE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-600 tracking-tight uppercase mb-1">
              Clinical Notes
            </label>
            <textarea
              rows="2"
              placeholder="Lesion presentation, itch, growth, bleeding symptoms..."
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              className="w-full px-3.5 py-2.5 bg-white border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition"
            ></textarea>
          </div>

          <div className="pt-2 flex justify-end gap-2">
            <Button
              type="button"
              variant="ghost"
              onClick={() => setOpenCreate(false)}
              disabled={creating}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              disabled={creating}
              className="shadow-sm shadow-indigo-100"
            >
              {creating ? 'Creating case...' : 'Create Case'}
            </Button>
          </div>
        </form>
      </Drawer>
    </div>
  );
}
