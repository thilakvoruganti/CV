import axios from "axios";

const baseURL = process.env.REACT_APP_API_BASE_URL || "https://cv-backend-rbjy.onrender.com";

export const api = axios.create({ baseURL });
