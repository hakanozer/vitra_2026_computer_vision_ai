import { iAllUser } from "../models/iAllUser"
import { apiConfig } from "./apiConfig"

export const allUser = () => {
    return apiConfig.get<iAllUser>('users')
}