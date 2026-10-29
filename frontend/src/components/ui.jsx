import React, { createContext, useContext, useState, useCallback } from 'react';

/* ------------------------------------------------------------------ */
/* Toast system                                                       */
/* ------------------------------------------------------------------ */

const ToastContext = createContext(null);

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return ctx;
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const toast = useCallback(
    ({ title, description, type = 'info', duration = 5000 } = {}) => {
      const id =
        typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).slice(2);
      setToasts((prev) => [...prev, { id, title, description, type, duration }]);
      const timer = setTimeout(
        () => setToasts((prev) => prev.filter((t) => t.id !== id)),
        duration
      );
      return () => clearTimeout(timer);
    },
    []
  );

  const dismiss = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toast, toasts, dismiss }}>
      {children}
      <ToastList toasts={toasts} setToasts={setToasts} />
    </ToastContext.Provider>
  );
}

const TYPE_STYLE = {
  success: {
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    icon: 'bg-emerald-100 text-emerald-700',
    dot: 'bg-emerald-500',
    title: 'text-emerald-800',
    glyph: '✔',
  },
  error: {
    bg: 'bg-rose-50',
    border: 'border-rose-200',
    icon: 'bg-rose-100 text-rose-700',
    dot: 'bg-rose-500',
    title: 'text-rose-800',
    glyph: '✖',
  },
  warn: {
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    icon: 'bg-amber-100 text-amber-700',
    dot: 'bg-amber-500',
    title: 'text-amber-800',
    glyph: '⚠',
  },
  info: {
    bg: 'bg-indigo-50',
    border: 'border-indigo-200',
    icon: 'bg-indigo-100 text-indigo-700',
    dot: 'bg-indigo-500',
    title: 'text-indigo-800',
    glyph: 'ℹ',
  },
};

function ToastItem({ item, setToasts }) {
  const s = TYPE_STYLE[item.type] ?? TYPE_STYLE.info;
  return (
    <div
      key={item.id}
      className={`flex items-center gap-3 w-full min-w-[340px] p-3.5 rounded-xl border shadow-lg ${s.bg} ${s.border} animate-[fadeIn_400ms_ease-out_both] pointer-events-auto`}
    >
      <div className={`shrink-0 ${s.icon} w-8 h-8 rounded-lg flex items-center justify-center text-xs`}>
        {s.glyph}
      </div>
      <div className="flex-1 min-w-0">
        {item.title && (
          <p className={`${s.title} text-xs font-bold leading-none`}>{item.title}</p>
        )}
        {item.description && (
          <p className="text-xs text-slate-600 leading-relaxed mt-0.5">{item.description}</p>
        )}
      </div>
      <button
        onClick={() => setToasts((prev) => prev.filter((t) => t.id !== item.id))}
        className="shrink-0 text-slate-400 hover:text-slate-600 transition"
        aria-label="Dismiss notification"
      >
        ✕
      </button>
    </div>
  );
}

function ToastList({ toasts, setToasts }) {
  return (
    <div className="fixed top-4 right-4 z-[60] flex flex-col gap-2 pointer-events-none">
      {toasts.map((item) => (
        <ToastItem key={item.id} item={item} setToasts={setToasts} />
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Button                                                             */
/* ------------------------------------------------------------------ */

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  disabled,
  loading,
  className = '',
  ...props
}) {
  const base =
    'inline-flex items-center justify-center gap-2 font-semibold rounded-lg transition-all duration-200 outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-60 disabled:pointer-events-none whitespace-nowrap';
  const variants = {
    primary:
      'bg-indigo-600 text-white hover:bg-indigo-700 focus-visible:ring-indigo-500 shadow-sm shadow-indigo-100',
    secondary:
      'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200 focus-visible:ring-slate-400',
    ghost:
      'bg-transparent text-slate-600 hover:bg-slate-100 focus-visible:ring-slate-400',
    danger:
      'bg-rose-600 text-white hover:bg-rose-700 focus-visible:ring-rose-500 shadow-sm shadow-rose-100',
    outline:
      'border border-slate-200 bg-transparent text-slate-700 hover:bg-slate-50 focus-visible:ring-slate-400',
  };
  const sizes = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2.5 text-sm',
    lg: 'px-6 py-3 text-base',
  };
  return (
    <button
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled}
      {...props}
    >
      {loading && (
        <span className="flex items-center gap-1">
          <span
            className="w-3 h-3 border-2 border-current border-white/60 rounded-full animate-spin-slow"
          />
        </span>
      )}
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* Input                                                              */
/* ------------------------------------------------------------------ */

export function Input({
  id,
  label,
  type = 'text',
  placeholder,
  value,
  onChange,
  required,
  disabled,
  error,
  helper,
  className = '',
  ...props
}) {
  return (
    <div className="space-y-2">
      {label && (
        <label
          htmlFor={id}
          className="block text-xs font-semibold text-slate-600 tracking-tight uppercase"
        >
          {label}
        </label>
      )}
      <input
        id={id}
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        required={required}
        disabled={disabled}
        className={`flex w-full items-center px-3.5 py-2.5 text-sm text-slate-900 bg-white border rounded-lg transition-all
          ${error
            ? 'border-rose-500 text-rose-800 focus:ring-2 focus:ring-rose-500'
            : 'border-slate-200 focus:ring-2 focus:ring-indigo-500'}
          ${disabled ? 'bg-slate-50 text-slate-400 cursor-not-allowed' : ''}
          ${className}`}
        {...props}
      />
      {error && <p className="text-[10px] text-rose-600 font-medium">{error}</p>}
      {helper && !error && <p className="text-[10px] text-slate-400">{helper}</p>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Card                                                               */
/* ------------------------------------------------------------------ */

export function Card({ children, className = '', padding = 'p-6', ...props }) {
  return (
    <div
      className={`bg-white border border-slate-200 rounded-2xl shadow-sm ${padding} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className = '' }) {
  return (
    <div className={`mb-5 flex items-center justify-between gap-2 ${className}`}>
      {children}
    </div>
  );
}

export function CardTitle({ children, className = '' }) {
  return (
    <h3 className={`text-sm font-bold text-slate-900 tracking-tight uppercase letter-spacing-[0.02em] ${className}`}>
      {children}
    </h3>
  );
}

export function CardBody({ children, className = '' }) {
  return <div className={className}>{children}</div>;
}

/* ------------------------------------------------------------------ */
/* Badge                                                              */
/* ------------------------------------------------------------------ */

export function Badge({ children, variant = 'neutral', className = '' }) {
  const styles = {
    neutral: 'bg-slate-100 text-slate-600 border border-slate-200',
    brand: 'bg-indigo-50 text-indigo-700 border border-indigo-100',
    success: 'bg-emerald-100 text-emerald-700 border border-emerald-200',
    warn: 'bg-amber-100 text-amber-700 border border-amber-200',
    error: 'bg-rose-100 text-rose-700 border border-rose-200',
    critical: 'bg-rose-500 text-white',
    high: 'bg-orange-500 text-white',
    moderate: 'bg-amber-500 text-slate-900',
    low: 'bg-emerald-600 text-white',
    outline: 'border border-slate-300 bg-transparent text-slate-500',
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded-full border shadow-xs ${styles[variant]} ${className}`}
    >
      {children}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Drawer (side sheet)                                                */
/* ------------------------------------------------------------------ */

export function Drawer({ isOpen, onClose, title, className = '', children }) {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-[50]">
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        className={`absolute right-0 top-0 h-full w-full max-w-[480px] bg-white border-l border-slate-200 shadow-2xl flex flex-col z-[51] animate-[slideUp_350ms_ease-out_both] ${className}`}
      >
        <div className="px-6 pb-3 border-b border-slate-100 flex items-center justify-between shrink-0">
          <h3 className="text-base font-bold text-slate-900">{title}</h3>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 transition shrink-0"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="p-6 overflow-y-auto flex-1">{children}</div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Modal (centered dialog)                                            */
/* ------------------------------------------------------------------ */

export function Modal({ isOpen, onClose, className = '', children }) {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-[50] flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/45 backdrop-blur-md"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        className={`relative bg-white rounded-3xl shadow-2xl w-full max-w-lg max-h-[90vh] flex flex-col animate-[fadeIn_300ms_ease-out_both] ${className}`}
      >
        <div className="px-6 pb-3 border-b border-slate-100 flex items-center justify-between shrink-0">
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 transition shrink-0"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="p-6 overflow-y-auto">{children}</div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Skeleton                                                           */
/* ------------------------------------------------------------------ */

export function Skeleton({
  className = '',
  width = '100%',
  height = '16px',
  rounded = 'rounded-lg',
}) {
  return (
    <div
      className={`animate-pulse-soft bg-slate-200 ${rounded} w-full h-full ${className}`}
      style={{ width, height }}
    />
  );
}

/* ------------------------------------------------------------------ */
/* Donut chart                                                        */
/* ------------------------------------------------------------------ */

const DONUT_COLORS = {
  indigo: '#4f46e5',
  emerald: '#10b981',
  amber: '#f59e0b',
  rose: '#f43f5e',
  slate: '#64748b',
};

export function Donut({
  pct,
  size = 96,
  stroke = 10,
  color = 'indigo',
  label,
  className = '',
}) {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;
  const strokeColor = DONUT_COLORS[color] ?? color;
  return (
    <div
      className={`relative flex items-center justify-center w-[${size}px] h-[${size}px] ${className}`}
    >
      <svg className="w-full h-full transform -rotate-90 overflow-visible">
        <circle
          cx="50%"
          cy="50%"
          r={radius}
          stroke="rgb(226 232 240)"
          strokeWidth={stroke}
          fill="none"
        />
        <circle
          cx="50%"
          cy="50%"
          r={radius}
          stroke={strokeColor}
          strokeWidth={stroke}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-700 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {label && (
          <span className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">
            {label}
          </span>
        )}
        <span className="text-sm font-black text-slate-900">{pct}%</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Section heading                                                    */
/* ------------------------------------------------------------------ */

export function Section({ title, subtitle, className = '' }) {
  return (
    <div className={`mb-6 space-y-1 ${className}`}>
      <h2 className="text-lg font-bold text-slate-900 tracking-tight">{title}</h2>
      {subtitle && (
        <p className="text-sm text-slate-500">{subtitle}</p>
      )}
    </div>
  );
}
