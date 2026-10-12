import { apiGet, apiPost } from "./client";

export const register = (name, email, password) =>
  apiPost("/api/auth/register", { name, email, password });

export const login = (email, password) =>
  apiPost("/api/auth/login", { email, password });

export const logout = () => apiPost("/api/auth/logout", {});

export const fetchMe = () => apiGet("/api/auth/me");

export const saveTopics = (topics) => apiPost("/api/auth/topics", { topics });
