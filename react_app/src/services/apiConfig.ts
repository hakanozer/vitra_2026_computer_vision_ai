import axios from "axios";

export const apiConfig = axios.create({
    baseURL: 'https://jsonbulut.com/api/',
    timeout: 15000
})