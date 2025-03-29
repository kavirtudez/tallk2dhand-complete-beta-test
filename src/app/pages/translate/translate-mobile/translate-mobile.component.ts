import {Component, inject} from '@angular/core';
import {BaseComponent} from '../../../components/base/base.component';
import {IonContent, IonFooter} from '@ionic/angular/standalone';
import {SpokenLanguageInputComponent} from '../spoken-to-signed/spoken-language-input/spoken-language-input.component';
import {SignedLanguageOutputComponent} from '../spoken-to-signed/signed-language-output/signed-language-output.component';
import {Store} from '@ngxs/store';
import {SetSpokenLanguage, SetSignedLanguage} from '../../../modules/translate/translate.actions';
import {SetSetting} from '../../../modules/settings/settings.actions';

@Component({
  selector: 'app-translate-mobile',
  templateUrl: './translate-mobile.component.html',
  styleUrls: ['./translate-mobile.component.scss'],
  imports: [
    IonContent,
    IonFooter,
    SignedLanguageOutputComponent,
    SpokenLanguageInputComponent
  ],
})
export class TranslateMobileComponent extends BaseComponent {
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
