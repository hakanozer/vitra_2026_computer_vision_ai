export interface iUser {
  meta: Meta
  data: Data
}

export interface iProfile {
  meta: Meta
  data: User
}

export interface Meta {
  status: number
  message: string
}

export interface Data {
  access_token: string
  token_type: string
  expires_in: number
  user: User
}

export interface User {
  id: number
  name: string
  email: string
  role: string
  remember_token: any
  created_at: string
  updated_at: string
}
