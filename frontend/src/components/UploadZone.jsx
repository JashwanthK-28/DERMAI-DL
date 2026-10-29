import React, { useState, useRef } from 'react';
import { imageApi } from '../api/client';
import { Button, Badge } from '../components/ui';

export default function UploadZone({ onUploadSuccess, caseId }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUploadResult, setLastUploadResult] = useState(null);
  const inputRef = useRef(null);

  const validateFile = (file) => {
    setError(null);
    if (!file) return false;

    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setError(`Invalid file format '${file.type}'. Please upload JPEG or PNG images.`);
      return false;
    }

    const maxSize = 15 * 1024 * 1024;
    if (file.size > maxSize) {
      setError(`File size (${(file.size / (1024 * 1024)).toFixed(1)}MB) exceeds maximum limit of 15MB.`);
      return false;
    }

    if (file.size === 0) {
      setError('Empty file rejected. Zero-byte payloads are invalid.');
      return false;
    }

    return true;
  };

  const handleFile = (file) => {
    if (validateFile(file)) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setLastUploadResult(null);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile || !caseId) return;
    setUploading(true);
    setError(null);

    try {
      const data = await imageApi.upload(caseId, selectedFile);
      setLastUploadResult(data);
      if (onUploadSuccess) {
        onUploadSuccess(data.image);
      }
    } catch (err) {
      setError(err.message || 'Image upload and preprocessing failed.');
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    setLastUploadResult(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div className="space-y-4">
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-2xl p-6 text-center transition-all ${
          dragActive
            ? 'border-indigo-500 bg-indigo-50/60 scale-[1.01]'
            : 'border-slate-300 hover:border-indigo-400 bg-white'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png, image/jpeg, image/jpg"
          onChange={handleChange}
          className="hidden"
          id="lesion-image-input"
        />

        {!previewUrl ? (
          <label htmlFor="lesion-image-input" className="cursor-pointer flex flex-col items-center gap-3">
            <div className="w-14 h-14 bg-indigo-50 text-indigo-600 rounded-2xl flex items-center justify-center text-2xl shadow-sm shadow-indigo-100">
              📷
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-700">
                Click to browse or drag & drop lesion photo
              </p>
              <p className="text-xs text-slate-400 mt-1">
                Supports Smartphone PNG, JPG, JPEG (min. 100x100 resolution)
              </p>
            </div>
          </label>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-center gap-4">
              <div className="relative group">
                <img
                  src={previewUrl}
                  alt="Client Preview"
                  className="w-40 h-40 object-cover rounded-xl border border-slate-200 shadow-md"
                />
                <span className="absolute bottom-2 left-2 bg-black/70 text-white text-[10px] px-2 py-0.5 rounded backdrop-blur">
                  {(selectedFile.size / 1024).toFixed(0)} KB
                </span>
              </div>
            </div>

            <p className="text-xs text-slate-600 font-medium">{selectedFile.name}</p>

            <div className="flex justify-center gap-2">
              <button
                onClick={handleUploadSubmit}
                disabled={uploading}
                className="px-5 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold rounded-lg shadow-sm shadow-indigo-100 transition disabled:opacity-60 disabled:pointer-events-none"
              >
                {uploading ? 'Processing Pipeline...' : 'Upload & Preprocess'}
              </button>
              <button
                onClick={handleReset}
                disabled={uploading}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-medium rounded-lg transition disabled:opacity-60"
              >
                Remove
              </button>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-700 px-4 py-3 rounded-xl text-xs flex items-center justify-between">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)} className="text-rose-500 font-bold hover:text-rose-700">
            ✕
          </button>
        </div>
      )}

      {lastUploadResult && (
        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-bold text-slate-900">
              Post-Upload Preprocessing Comparison
            </h4>
            <Badge variant="success" className="text-[10px]">
              Standardized to 224x224
            </Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2 text-center">
              <p className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
                Original Smartphone Image ({lastUploadResult.preprocessing.original_dim})
              </p>
              <img
                src={`/api/v1/images/${lastUploadResult.image.id}/file`}
                alt="Original Lesion"
                className="w-full h-48 object-cover rounded-xl border border-slate-300 shadow-sm"
              />
            </div>
            <div className="space-y-2 text-center">
              <p className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
                Preprocessed Inference Tensor ({lastUploadResult.preprocessing.processed_dim})
              </p>
              <img
                src={`/api/v1/images/${lastUploadResult.image.id}/preprocessed_file`}
                alt="Preprocessed Lesion"
                className="w-full h-48 object-cover rounded-xl border border-indigo-300 shadow-sm"
              />
            </div>
          </div>

          <QualityBadge quality={lastUploadResult.preprocessing} />
        </div>
      )}
    </div>
  );
}
