import React from 'react';
import { Badge, Donut } from '../components/ui';

const CLASS_COLORS = {
  MEL: { bg: 'bg-rose-500', text: 'text-white', bar: 'bg-rose-500' },
  BCC: { bg: 'bg-orange-500', text: 'text-white', bar: 'bg-orange-500' },
  SCC: { bg: 'bg-amber-500', text: 'text-slate-900', bar: 'bg-amber-500' },
  ACK: { bg: 'bg-yellow-500', text: 'text-slate-900', bar: 'bg-yellow-500' },
  NEV: { bg: 'bg-emerald-500', text: 'text-white', bar: 'bg-emerald-500' },
  SEK: { bg: 'bg-teal-500', text: 'text-white', bar: 'bg-teal-500' },
};

const RISK_STYLE = {
  CRITICAL: { badge: 'critical', label: 'Critical' },
  HIGH: { badge: 'high', label: 'High' },
  MODERATE: { badge: 'moderate', label: 'Moderate' },
  LOW: { badge: 'low', label: 'Low' },
  default: { badge: 'neutral', label: 'Low' },
};

export default function PredictionCard({ prediction, onRunInference, loading }) {
  if (!prediction) {
    return (
      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm text-center space-y-4">
        <div className="w-14 h-14 bg-indigo-50 text-indigo-600 rounded-2xl flex items-center justify-center mx-auto text-2xl font-bold shadow-sm shadow-indigo-100">
          AI
        </div>
        <h4 className="text-base font-semibold text-slate-800">No Inference Results Yet</h4>
        <p className="text-sm text-slate-500 max-w-sm mx-auto leading-relaxed">
          Pass the standardized lesion image through the deep learning classifier to
          generate diagnostic probabilities and risk assessment.
        </p>
        <div className="flex flex-col sm:flex-row justify-center gap-2 pt-2">
          <button
            onClick={() => onRunInference('resnet50_baseline')}
            disabled={loading}
            className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-lg shadow-sm shadow-indigo-100 transition disabled:opacity-60 disabled:pointer-events-none"
          >
            {loading ? 'Evaluating...' : 'Run ResNet50'}
          </button>
          <button
            onClick={() => onRunInference('efficientnet_b0')}
            disabled={loading}
            className="px-4 py-2.5 bg-slate-800 hover:bg-slate-900 text-white text-sm font-medium rounded-lg shadow-sm transition disabled:opacity-60 disabled:pointer-events-none"
          >
            {loading ? 'Evaluating...' : 'Run EfficientNet-B0'}
          </button>
        </div>
      </div>
    );
  }

  const {
    predicted_class,
    predicted_description,
    confidence_score,
    class_probabilities,
    risk_category,
    model_name,
    model_version,
  } = prediction;

  const confidencePct = (confidence_score * 100).toFixed(1);
  const risk = RISK_STYLE[risk_category?.toUpperCase()] ?? RISK_STYLE.default;
  const style = CLASS_COLORS[predicted_class] ?? CLASS_COLORS.NEV;

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-6">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <div>
          <span className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">
            Model Inference Result
          </span>
          <p className="text-xs text-slate-500">{model_name} (v{model_version})</p>
        </div>
        <Badge variant={risk.badge} className="text-[10px] font-bold shadow-sm">
          Risk: {risk.label}
        </Badge>
      </div>

      {/* Headline result */}
      <div className="bg-slate-50 p-5 rounded-xl border border-slate-100 flex items-center justify-between gap-4">
        <div className="flex-1 min-w-0">
          <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">
            Predicted Diagnosis
          </p>
          <h3 className="text-xl font-bold text-slate-900 flex items-baseline gap-2 mt-0.5">
            <span className={`${style.bg} ${style.text} px-2 py-0.5 rounded-md text-[10px] font-bold`}>
              {predicted_class}
            </span>
            <span className="text-xs font-normal text-slate-500">
              ({predicted_description})
            </span>
          </h3>
        </div>
        <div className="flex flex-col items-end gap-1">
          <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">
            Confidence
          </p>
          <Donut pct={confidencePct} size="64" stroke="8" color="indigo" label="Confidence" />
        </div>
      </div>

      {/* Probabilities */}
      <div className="space-y-3">
        <p className="text-xs font-bold uppercase tracking-wider text-slate-500">
          Class Probability Distribution
        </p>
        <div className="space-y-2.5">
          {class_probabilities &&
            Object.entries(class_probabilities).map(([cls, prob]) => {
              const pct = (prob * 100).toFixed(1);
              const isSelected = cls === predicted_class;
              const c = CLASS_COLORS[cls] ?? CLASS_COLORS.NEV;
              return (
                <div key={cls} className="space-y-1">
                  <div className="flex justify-between text-xs font-medium">
                    <span
                      className={`flex items-center gap-1.5 ${isSelected ? 'font-bold text-slate-900' : 'text-slate-600'}`}
                    >
                      {cls}
                      {isSelected && <span className="text-[9px]">★</span>}
                    </span>
                    <span className={isSelected ? 'font-bold text-indigo-600' : 'text-slate-500'}>
                      {pct}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-2 rounded-full transition-all duration-700 ease-out ${c.bar}`}
                      style={{ width: `${pct}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      <div className="pt-3 border-t border-slate-100 flex justify-end gap-2">
        <button
          onClick={() => onRunInference('resnet50_baseline')}
          disabled={loading}
          className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium rounded-lg transition disabled:opacity-60"
        >
          Re-run ResNet50
        </button>
        <button
          onClick={() => onRunInference('efficientnet_b0')}
          disabled={loading}
          className="px-3 py-1.5 bg-slate-800 hover:bg-slate-900 text-white text-xs font-medium rounded-lg transition disabled:opacity-60"
        >
          Run EfficientNet-B0
        </button>
      </div>
    </div>
  );
}
