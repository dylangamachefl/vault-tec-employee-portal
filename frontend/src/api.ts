import { User, Document, QueryResult } from './types';

const BASE = '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export function login(userId: string): Promise<User> {
  return request<User>('/login', {
    method: 'POST',
    body: JSON.stringify({ user_id: userId }),
  });
}

export function getDocuments(accessLevel: string): Promise<Document[]> {
  return request<Document[]>(
    `/documents?access_level=${encodeURIComponent(accessLevel)}`
  );
}

export function sendQuery(
  query: string,
  accessLevel: string,
  topK: number = 5
): Promise<QueryResult> {
  return request<QueryResult>('/query', {
    method: 'POST',
    body: JSON.stringify({ query, access_level: accessLevel, top_k: topK }),
  });
}
