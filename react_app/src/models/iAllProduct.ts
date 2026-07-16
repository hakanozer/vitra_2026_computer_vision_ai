import { Meta as iUserMeta } from "./iUser"

export interface iAllProduct {
  meta: Meta
  data: iProduct[]
}

export interface iSingleProduct {
  meta: iUserMeta
  data: iProduct
}

export interface Meta {
  status: number
  message: string
  pagination: Pagination
}

export interface Pagination {
  page: number
  per_page: number
  total_items: number
  total_pages: number
}

export interface iProduct {
  id: number
  title: string
  description: string
  category: string
  price: number
  discountPercentage: number
  rating: number
  stock: number
  tags: string[]
  brand: string
  sku: string
  minimumOrderQuantity: number
  images: string[]
}
