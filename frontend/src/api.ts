import { jwtDecode } from 'jwt-decode';
import { User, Document, QueryResult, AccessLevel } from './types';

// In production set VITE_API_URL=https://your-backend.onrender.com/api
// Locally, Vite's proxy forwards /api → localhost:8000, so the default works.
const BASE = import.meta.env.VITE_API_URL ?? '/api';

export interface JwtPayload {
  sub: string;
  username: string;
  role: string;
  allowed_levels: string[];
  exp: number;
  iat: number;
}

function getToken(): string | null {
  return localStorage.getItem('vault_token');
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  const token = getToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE}${url}`, {
    ...options,
    headers: { ...headers, ...options?.headers },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function login(userId: string): Promise<User> {
  const data = await request<{ access_token: string }>('/login', {
    method: 'POST',
    body: JSON.stringify({ user_id: userId }),
  });

  localStorage.setItem('vault_token', data.access_token);

  const payload = jwtDecode<JwtPayload>(data.access_token);
  if (!payload) throw new Error('Invalid token received');

  return {
    id: payload.sub,
    username: payload.username,
    role: payload.role,
    accessLevel: (payload.allowed_levels?.[0] ?? 'General') as AccessLevel,
  };
}

export function logout(): void {
  localStorage.removeItem('vault_token');
}

export function getDocuments(): Promise<Document[]> {
  return request<Document[]>('/documents');
}

export function sendQuery(
  query: string,
  topK: number = 5,
  vector?: number[],
): Promise<QueryResult> {
  return request<QueryResult>('/query', {
    method: 'POST',
    body: JSON.stringify({ query, top_k: topK, ...(vector ? { vector } : {}) }),
  });
}

export function getDocumentContent(docId: string): Promise<{ content: string }> {
  return request<{ content: string }>(`/documents/${docId}/content`);
}
