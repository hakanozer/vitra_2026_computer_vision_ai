import { Meta } from "./iUser"

export interface iAllUser {
  meta: Meta
  data: SingleUser[]
}

export interface SingleUser {
  id: number
  name: string
  username: string
  email: string
  password: string
  role: string
  profile: string
  address: Address
  phone: string
  website: string
  company: Company
}

export interface Address {
  street: string
  suite: string
  city: string
  zipcode: string
  geo: Geo
}

export interface Geo {
  lat: string
  lng: string
}

export interface Company {
  name: string
  catchPhrase: string
  bs: string
}
