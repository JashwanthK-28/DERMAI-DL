import React, { useState, useEffect } from 'react';
import { casesApi, imageApi, inferenceApi } from '../api/client';
import UploadZone from '../components/UploadZone';
import QualityBadge from '../components/QualityBadge';
import PredictionCard from '../components/PredictionCard';
import { Button, Badge, Card, CardHeader, CardTitle, CardBody, Donut } from '../components/ui';

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

function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

export default function CaseDetails({ caseItem, onBack }) {
  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState(true);
  const [inferring, setInferring] = useState(false);

  const loadCaseImages = async () => {
    try {
      setLoading(true);
      const data = await imageApi.list(caseItem.id);
      setImages(data);
      if (data.length > 0 && !selectedImage) {
        setSelectedImage(data[0]);
      }
    } catch (err) {
      console.error('Failed to load images:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCaseImages();
  }, [caseItem.id]);

  useEffect(() => {
    if (selectedImage) {
      loadPredictions(selectedImage.id);
    }
  }, [selectedImage]);

  const loadPredictions = async (imageId) => {
    try {
      const preds = await inferenceApi.getPredictions(imageId);
      if (preds && preds.length > 0) {
        setPredictions((prev) => ({ ...prev, [imageId]: preds[0] }));
      }
    } catch (err) {
      console.error('Failed to load predictions:', err);
    }
  };

  const handleUploadSuccess = (newImage) => {
    setImages((prev) => [newImage, ...prev]);
    setSelectedImage(newImage);
  };

  const handleRunInference = async (modelName = 'resnet50_baseline') => {
    if (!selectedImage) return;
    setInferring(true);
    try {
      const pred = await inferenceApi.predict(selectedImage.id, modelName);
      setPredictions((prev) => ({ ...prev, [selectedImage.id]: pred }));
    } catch (err) {
      alert(err.response?.data?.detail || 'Inference failed');
    } finally {
      setInferring(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onBack}
              className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold rounded-lg transition flex items-center gap-1.5"
            >
              <span>←</span>
              <span>Back to Dashboard</span>
            </button>
            <div className="h-5 w-px bg-slate-200"></div>
            <div>
              <h1 className="text-sm font-bold text-slate-900">
                Case: {caseItem.patient_clinical_id || caseItem.id.slice(0, 8)}
              </h1>
              <p className="text-[11px] text-slate-400">
                {SITE_LABELS[caseItem.anatomic_site] || caseItem.anatomic_site} • Age:
                {caseItem.age ?? 'N/A'} • {caseItem.sex || 'N/A'}
              </p>
            </div>
          </div>

          <Badge
            variant="brand"
            className="text-[10px] uppercase tracking-wider font-bold"
          >
            {SITE_LABELS[caseItem.anatomic_site] || 'Dermatology Case'}
          </Badge>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full space-y-6">
        {/* Upload zone */}
        <Card className="shadow-sm border-slate-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span>📷</span>
              Mobile Image Capture & Preprocessing
            </CardTitle>
          </CardHeader>
          <CardBody className="p-0 pt-4 space-y-4">
            <UploadZone
              caseId={caseItem.id}
              onUploadSuccess={handleUploadSuccess}
            />
          </CardBody>
        </Card>

        {/* Gallery + analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: thumbnails */}
          <div className="lg:col-span-4 space-y-4">
            <Card className="shadow-sm border-slate-200">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center justify-between">
                  <span>Case Images</span>
                  <span className="text-[10px] text-slate-400 font-medium">
                    {images.length} uploaded
                  </span>
                </CardTitle>
              </CardHeader>
              <CardBody className="p-0 pt-3 space-y-3">
                {loading ? (
                  <div className="p-6 text-center">
                    <div className="flex items-center justify-center gap-2 text-xs text-slate-400 mb-4">
                      <span className="w-3 h-3 border-2 border-current border-slate-400 rounded-full animate-spin-slow" />
                      Loading images...
                    </div>
                    <Skeleton width="100%" height="72px" className="mx-auto w-24" />
                  </div>
                ) : images.length === 0 ? (
                  <div className="p-6 text-center space-y-2">
                    <div className="w-10 h-10 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center mx-auto text-lg">
                      📷
                    </div>
                    <p className="text-xs text-slate-400">
                      No images uploaded for this case yet.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-[520px] overflow-y-auto pr-2">
                    {images.map((img) => {
                      const isSelected = selectedImage?.id === img.id;
                      return (
                        <div
                          key={img.id}
                          onClick={() => setSelectedImage(img)}
                          className={`flex items-center gap-3 p-2.5 rounded-xl border transition cursor-pointer ${
                            isSelected
                              ? 'border-indigo-600 bg-indigo-50/60 shadow-sm'
                              : 'border-slate-200 hover:border-slate-300 bg-white'
                          }`}
                        >
                          <img
                            src={`/api/v1/images/${img.id}/preprocessed_file`}
                            alt={img.original_filename}
                            className="w-12 h-12 object-cover rounded-xl border border-slate-200"
                          />
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-bold text-slate-800 truncate">
                              {img.original_filename}
                            </p>
                            <p className="text-[10px] text-slate-400">
                              {img.width}x{img.height} • {(img.file_size_bytes / 1024).toFixed(0)} KB
                            </p>
                            {img.quality && (
                              <span className="inline-block mt-1 text-[9px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700">
                                {img.quality.quality_status}
                              </span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </CardBody>
            </Card>
          </div>

          {/* Right: analysis */}
          <div className="lg:col-span-8 space-y-6">
            {selectedImage ? (
              <div className="space-y-6">
                {/* Comparison */}
                <Card className="shadow-sm border-slate-200">
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center justify-between">
                      <span>Preprocessing Pipeline Verification</span>
                      <Badge variant="brand" className="text-[10px]">
                        8-Step Engine
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardBody className="p-0 pt-4 space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-2 text-center">
                        <p className="text-xs font-bold text-slate-600 uppercase tracking-wider">
                          Original Smartphone Image
                        </p>
                        <div className="aspect-square bg-slate-100 rounded-2xl overflow-hidden border border-slate-200">
                          <img
                            src={`/api/v1/images/${selectedImage.id}/file`}
                            alt="Original Lesion"
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <p className="text-[11px] text-slate-400">
                          {selectedImage.width}x{selectedImage.height} raw resolution
                        </p>
                      </div>
                      <div className="space-y-2 text-center">
                        <p className="text-xs font-bold text-indigo-600 uppercase tracking-wider">
                          Model Tensor (224x224)
                        </p>
                        <div className="aspect-square bg-indigo-50 rounded-2xl overflow-hidden border border-indigo-200 shadow-sm">
                          <img
                            src={`/api/v1/images/${selectedImage.id}/preprocessed_file`}
                            alt="Preprocessed Lesion"
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <p className="text-[11px] text-indigo-500">
                          Bicubic + Float32 [0.0, 1.0]
                        </p>
                      </div>
                    </div>
                    <QualityBadge quality={selectedImage.quality} />
                  </CardBody>
                </Card>

                {/* Prediction */}
                <PredictionCard
                  prediction={predictions[selectedImage.id]}
                  onRunInference={handleRunInference}
                  loading={inferring}
                />
              </div>
            ) : (
              <Card className="text-center py-16 border-dashed border-slate-300 hover:border-indigo-400 transition">
                <div className="w-14 h-14 bg-slate-100 text-slate-400 rounded-2xl flex items-center justify-center mx-auto text-2xl mb-3">
                  📷
                </div>
                <h4 className="text-base font-semibold text-slate-800">Select an Image</h4>
                <p className="text-xs text-slate-500 max-w-sm mx-auto mt-2">
                  Upload or select a lesion image to view preprocessing comparison and
                  deep learning predictions.
                </p>
              </Card>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
