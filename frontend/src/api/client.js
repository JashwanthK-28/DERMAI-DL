import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export const authApi = {
  login: async (email, password) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    const res = await api.post('/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
  register: async (email, password, role = 'clinician') => {
    const res = await api.post('/auth/register', { email, password, role });
    return res.data;
  },
  getMe: async () => {
    const res = await api.get('/auth/me');
    return res.data;
  },
};

export const casesApi = {
  list: async () => {
    const res = await api.get('/cases');
    return res.data;
  },
  get: async (id) => {
    const res = await api.get(`/cases/${id}`);
    return res.data;
  },
  create: async (caseData) => {
    const res = await api.post('/cases', caseData);
    return res.data;
  },
  update: async (id, caseData) => {
    const res = await api.put(`/cases/${id}`, caseData);
    return res.data;
  },
  delete: async (id) => {
    await api.delete(`/cases/${id}`);
  },
};

export const imageApi = {
  upload: async (caseId, file) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post(`/cases/${caseId}/images/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
  list: async (caseId) => {
    const res = await api.get(`/cases/${caseId}/images`);
    return res.data;
  },
  getOriginalUrl: (id) => `/api/v1/images/${id}/file`,
  getPreprocessedUrl: (id) => `/api/v1/images/${id}/preprocessed_file`,
};

export const inferenceApi = {
  predict: async (imageId, modelName = 'resnet50_baseline') => {
    const res = await api.post(`/images/${imageId}/predict`, { model_name: modelName });
    return res.data;
  },
  getPredictions: async (imageId) => {
    const res = await api.get(`/images/${imageId}/predictions`);
    return res.data;
  },
};

export default api;
