import { Injectable } from '@angular/core';
import {HttpClient, HttpHeaders, HttpErrorResponse} from '@angular/common/http';
import {Observable, throwError} from 'rxjs';
import {API_URL} from '../environments/environment';

const httpOptions = {
  headers: new Headers({
    'Access-Control-Allow-Origin':'*',
    'Auth':'authkey',
    'userid':'1'
  })
}

@Injectable({
  providedIn: 'root'
})


export class ApiService {

  constructor(private http: HttpClient) { }

  private getHeaders(): HttpHeaders {
    return new HttpHeaders()
        .set('Content-Type', 'application/json')
        .set('Access-Control-Allow-Origin','*')
        .set("Access-Control-Allow-Methods", "DELETE, POST, GET, OPTIONS")
        .set("Access-Control-Allow-Headers", "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With")
      }

  postCargaTweets(params?: any): Observable<any> {
    try {
      return this.http.post(`${API_URL}/cargarTweets`, params, { headers: this.getHeaders() });
    } catch (error) {
      return throwError(new Error("Oops!"))
    }
  }

  postGenerarDataset(params?: any): Observable<any> {
    try {
      var args = params == null ? {} : params;
      return this.http.post(`${API_URL}/generarDataset`, args, { headers: this.getHeaders() });
    } catch (error) {
      return throwError(new Error("Oops!"))
    }
  }

  postTrainModels(params?: any): Observable<any> {
    try {
      var args = params == null ? {} : params;
      return this.http.post(`${API_URL}/trainModels`, args, { headers: this.getHeaders() });
    } catch (error) {
      return throwError(new Error("Oops!"))
    }
  }

  postTrainEnsambleStacking(params?: any): Observable<any> {
    try {
      var args = params == null ? {} : params;
      return this.http.post(`${API_URL}/trainEnsambleStacking`, args, { headers: this.getHeaders() });
    } catch (error) {
      return throwError(new Error("Oops!"))
    }
  }

  postTrainEnsamblePromedio(params?: any): Observable<any> {
    try {
      var args = params == null ? {} : params;
      return this.http.post(`${API_URL}/trainEnsamblePromedio`, args, { headers: this.getHeaders() });
    } catch (error) {
      return throwError(new Error("Oops!"))
    }
  }

  postTrainEnsambleBagging(params?: any): Observable<any> {
    try {
      var args = params == null ? {} : params;
      return this.http.post(`${API_URL}/trainEnsambleBagging`, args, { headers: this.getHeaders() });
    } catch (error) {
      return throwError(new Error("Oops!"))
    }
  }
  postImage(params?: any): Observable<Blob> {
    try {
      var args = params == null ? {} : params;
      return this.http.post(`${API_URL}/postImage`, args, { headers: this.getHeaders() , responseType: 'blob'});
    } catch (error) {
      return throwError(new Error("Oops!"))
    }
  }

}
