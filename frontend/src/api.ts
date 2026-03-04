import { User, Document, QueryResult } from './types';

const BASE = '/api';

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

function parseJwt(token: string) {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(window.atob(base64).split('').map(function (c) {
      return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
    }).join(''));
    return JSON.parse(jsonPayload);
  } catch (e) {
    return null;
  }
}

export async function login(userId: string): Promise<User> {
  const data = await request<{ access_token: string }>('/login', {
    method: 'POST',
    body: JSON.stringify({ user_id: userId }),
  });

  localStorage.setItem('vault_token', data.access_token);

  const payload = parseJwt(data.access_token);
  if (!payload) throw new Error("Invalid token received");

  return {
    id: payload.sub,
    username: payload.username,
    role: payload.role,
    accessLevel: payload.allowed_levels?.[0] || 'General'
  };
}

export function logout(): void {
  localStorage.removeItem('vault_token');
}

export function getDocuments(): Promise<Document[]> {
  // Backend relies on JWT for access level now, query param not needed
  return request<Document[]>('/documents');
}

export function sendQuery(
  query: string,
  topK: number = 5
): Promise<QueryResult> {
  // Backend relies on JWT for access level now
  return request<QueryResult>('/query', {
    method: 'POST',
    body: JSON.stringify({ query, top_k: topK }),
  });
}
