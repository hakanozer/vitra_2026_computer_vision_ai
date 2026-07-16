import { iAllProduct, iSingleProduct } from "../models/iAllProduct"
import { apiConfig } from "./apiConfig"

// Query string
export const allProduct = (page: number) => {
    const sendObj = {
        page: page,
        per_page: 10
    }
    return apiConfig.get<iAllProduct>('products', {params: sendObj})
}

export const singleProduct = (id: string) => {
    return apiConfig.get<iSingleProduct>('products/'+id)
}

export const searchProduct = (q:string) => {
    const sendObj = {
        query: q,
        page: 1,
        per_page: 10
    }
    return apiConfig.get<iAllProduct>('products/search', {params: sendObj})
}

// Body -> apiConfig.post<iUser>('auth/login', sendObj)
// QueryString -> apiConfig.get('products', {params: sendObj})
// Path Variable -> apiConfig.get('products/'+id)