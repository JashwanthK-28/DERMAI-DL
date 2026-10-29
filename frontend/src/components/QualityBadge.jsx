import React from 'react';
import { Badge } from '../components/ui';

const STATUS_STYLE = {
  PASS: { badge: 'success', label: 'Passed' },
  WARNING: { badge: 'warn', label: 'Warning' },
  REJECT: { badge: 'error', label: 'Rejected' },
  default: { badge: 'neutral', label: 'Review' },
};

export default function QualityBadge({ quality }) {
  if (!quality) return null;

  const {
    blur_score,
    brightness_score,
    contrast_score,
    quality_status,
    quality_reasons,
    original_dim,
    processed_dim,
  } = quality;

  const status = STATUS_STYLE[quality_status?.toUpperCase()] ?? STATUS_STYLE.default;

  const metrics = [
    {
      label: 'Blur (Laplacian)',
      value: blur_score,
      hint: '&gt; 100 is sharp',
    },
    {
      label: 'Brightness',
      value: brightness_score,
      hint: '50 - 200 normal',
    },
    {
      label: 'Contrast',
      value: contrast_score,
      hint: '&gt; 20 adequate',
    },
  ];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
          <h4 className="text-sm font-semibold text-slate-800">
            Image Quality Assessment
          </h4>
        </div>
        <Badge variant={status.badge} className="text-[10px] font-bold shadow-xs">
          {status.label}
        </Badge>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {metrics.map((m, idx) => (
          <div key={idx} className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
            <p className="text-[9px] uppercase tracking-wider text-slate-400 font-medium">
              {m.label}
            </p>
            <p className="text-sm font-bold text-slate-800 mt-0.5">{m.value}</p>
            <p className="text-[9px] text-slate-400 mt-0.5">{m.hint}</p>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between text-xs text-slate-500 pt-1">
        <span>Original: <strong className="text-slate-700">{original_dim}</strong></span>
        <span>Standardized: <strong className="text-slate-700">{processed_dim}</strong></span>
      </div>

      {quality_reasons && quality_reasons.length > 0 && (
        <div className="bg-slate-50 rounded-xl p-3 text-xs text-slate-600 space-y-1.5 border border-slate-100">
          <p className="font-medium text-slate-700 text-[10px] uppercase tracking-wider flex items-center gap-1.5">
            <span className="w-1 h-1 rounded-full bg-slate-400"></span>
            Quality Diagnostics
          </p>
          {quality_reasons.map((reason, idx) => (
            <p key={idx} className="flex items-start gap-1.5 text-slate-600 leading-relaxed">
              <span className="text-slate-400 flex-shrink-0">•</span>
              <span>{reason}</span>
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
