import axios from "axios";

const RENDER_BASE_FALLBACK = "https://cv-backend-rbjy.onrender.com";

const detectBaseUrl = (): string => {
	const envUrl = process.env.REACT_APP_API_BASE_URL?.trim();
	if (envUrl) {
		return envUrl;
	}

	if (typeof window !== "undefined") {
		const origin = window.location.origin;
		const host = window.location.hostname;
		if (/github\.io$/i.test(host)) {
			return RENDER_BASE_FALLBACK;
		}
		const githubMatch = origin.match(/^(https?:\/\/[a-z0-9-]+)-(\d+)(\.app\.github\.dev)$/i);
		if (githubMatch) {
			return `${githubMatch[1]}-8000${githubMatch[3]}`;
		}

		if (/localhost|127\.0\.0\.1/.test(origin)) {
			return origin.replace(/:\d+$/, ":8000");
		}

		return origin;
	}

	return RENDER_BASE_FALLBACK;
};

const normalizeBaseUrl = (url: string): string => url.replace(/\/+$/, "");

export const apiBaseUrl = normalizeBaseUrl(detectBaseUrl());

export const resolveApiUrl = (path: string): string => {
	if (!path) {
		return apiBaseUrl;
	}
	if (/^https?:\/\//i.test(path)) {
		return path;
	}
	try {
		return new URL(path, `${apiBaseUrl}/`).toString();
	} catch {
		return `${apiBaseUrl}${path.startsWith("/") ? "" : "/"}${path}`;
	}
};

export const api = axios.create({ baseURL: apiBaseUrl });

const summarizePayload = (data: unknown): string => {
	if (typeof FormData !== "undefined" && data instanceof FormData) {
		const entries: string[] = [];
		data.forEach((value, key) => {
			if (value instanceof File) {
				entries.push(`${key}: [File ${value.name} · ${value.size} bytes]`);
			} else {
				entries.push(`${key}: ${String(value)}`);
			}
		});
		return `FormData { ${entries.join(", ")} }`;
	}
	if (data === undefined || data === null) {
		return "<empty>";
	}
	if (typeof data === "object") {
		try {
			return JSON.stringify(data);
		} catch {
			return "<unserializable object>";
		}
	}
	return String(data);
};

api.interceptors.request.use((config) => {
	if (typeof window !== "undefined" && !config.headers?.["x-api-logged"]) {
		const method = (config.method ?? "get").toUpperCase();
		const url = resolveApiUrl(config.url ?? "");
		const params = config.params ? ` params=${JSON.stringify(config.params)}` : "";
		// eslint-disable-next-line no-console -- intentional developer telemetry for debugging prod issues
		console.info(`[API] ${method} ${url}${params}`, summarizePayload(config.data));
	}
	return config;
});
