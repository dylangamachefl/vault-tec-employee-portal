export type AccessLevel = 'General' | 'HR' | 'Marketing' | 'Admin';

export interface User {
  id: string;
  username: string;
  role: string;
  accessLevel: AccessLevel;
}

export interface Document {
  id: string;
  title: string;
  department: string;
  accessLevel: string;
  status: string;
  effectiveDate: string;
}

/** Matches SourceCitation from src/pipelines/retrieval_chain.py */
export interface Citation {
  source_document: string;
  section_title: string | null;
  access_level: string;
  department: string;
  doc_date: string | null;
  doc_status: string | null;
}

export interface QueryResult {
  answer: string;
  sources: Citation[];
  retrieved_chunk_count: number;
  query: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  citations?: Citation[];
  status?: 'success' | 'warning' | 'error' | 'info';
}
