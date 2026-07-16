import { iProfile, iUser } from "../models/iUser"
import { apiConfig } from "./apiConfig"

export const userLogin = (email: string, password: string) => {
    const sendObj = {
        email: email,
        password: password
    }
    return apiConfig.post<iUser>('auth/login', sendObj)
}

export const userProfile = () => {
    return apiConfig.get<iProfile>('profile/me')
}

export const userLogout = () => {
    return apiConfig.post('auth/logout')
}