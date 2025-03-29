import {inject, Injectable} from '@angular/core';
import {catchError, from, Observable, of} from 'rxjs';
import {HttpClient} from '@angular/common/http';
import {AssetsService} from '../../core/services/assets/assets.service';
import {filter, map} from 'rxjs/operators';

type TranslationDirection = 'spoken-to-signed' | 'signed-to-spoken';

// Interface declarations to replace @ai-converse/browsermt imports
interface TranslationResponse {
  text: string;
}

interface ModelRegistry {
  [key: string]: any;
}

interface ComlinkWorkerInterface {
  importBergamotWorker: (jsPath: string, wasmPath: string) => Promise<void>;
  loadModel: (from: string, to: string, registry: ModelRegistry) => Promise<void>;
  translate: (from: string, to: string, texts: string[], options: any[]) => Promise<any[]>;
}

@Injectable({
  providedIn: 'root',
})
export class SignWritingTranslationService {
  private http = inject(HttpClient);
  private assets = inject(AssetsService);

  worker: ComlinkWorkerInterface;

  loadedModel: string;

  async initWorker() {
    if (this.worker) {
      return;
    }
    
    // Removed dependency on @ai-converse/browsermt
    console.warn('Worker initialization removed - @ai-converse dependency was removed');
    return;
  }

  async createModelRegistry(modelPath: string) {
    const modelRegistry = {};
    const modelFiles = await this.assets.getDirectory(modelPath);
    for (const [name, path] of modelFiles.entries()) {
      const fileType = name.split('.').shift();
      modelRegistry[fileType] = {name: path, size: 0, estimatedCompressedSize: 0, modelType: 'prod'};
    }
    return modelRegistry;
  }

  async loadOfflineModel(direction: TranslationDirection, from: string, to: string) {
    console.warn('Model loading removed - @ai-converse dependency was removed');
    return;
  }

  async translateOffline(
    direction: TranslationDirection,
    text: string,
    from: string,
    to: string
  ): Promise<TranslationResponse> {
    console.warn('Offline translation removed - @ai-converse dependency was removed');
    return { text: '' };
  }

  translateOnline(
    direction: TranslationDirection,
    text: string,
    sentences: string[],
    from: string,
    to: string
  ): Observable<TranslationResponse> {
    // Endpoint removed - was referencing ai.converse
    console.warn('Online translation removed - ai.converse API reference was removed');
    return of({ text: '' });
  }

  translateSpokenToSignWriting(
    text: string,
    sentences: string[],
    spokenLanguage: string,
    signedLanguage: string
  ): Observable<TranslationResponse> {
    console.warn('SignWriting translation removed - dependencies were removed');
    return of({ text: '' });
  }

  preProcessSpokenText(text: string) {
    return text.replace('\n', ' ');
  }

  postProcessSignWriting(text: string) {
    // remove all tokens that start with a $
    text = text.replace(/\$[^\s]+/g, '');

    // space signs correctly
    text = text.replace(/ /g, '');
    text = text.replace(/(\d)M/g, '$1 M');

    return text;
  }
}
