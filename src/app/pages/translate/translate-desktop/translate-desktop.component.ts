import {Component, inject} from '@angular/core';
import {Store} from '@ngxs/store';
import {SetSetting} from '../../../modules/settings/settings.actions';
import {BaseComponent} from '../../../components/base/base.component';
import {IonContent} from '@ionic/angular/standalone';
import {SpokenLanguageInputComponent} from '../spoken-to-signed/spoken-language-input/spoken-language-input.component';
import {SignedLanguageOutputComponent} from '../spoken-to-signed/signed-language-output/signed-language-output.component';
import {SetSpokenLanguage, SetSignedLanguage} from '../../../modules/translate/translate.actions';

@Component({
  selector: 'app-translate-desktop',
  templateUrl: './translate-desktop.component.html',
  styleUrls: ['./translate-desktop.component.scss'],
  imports: [
    IonContent,
    SpokenLanguageInputComponent,
    SignedLanguageOutputComponent
  ],
})
export class TranslateDesktopComponent extends BaseComponent {
  private store = inject(Store);

  constructor() {
    super();
    
    // Set spoken language to English
    this.store.dispatch(new SetSpokenLanguage('en'));
    
    // Set signed language to ASL
    this.store.dispatch(new SetSignedLanguage('ase'));
    
    // Enable necessary settings for the translator
    this.store.dispatch([
      new SetSetting('receiveVideo', true),
      new SetSetting('drawPose', true),
    ]);
  }
}
