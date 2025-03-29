import {Component, inject} from '@angular/core';
import {NavigationEnd, Router} from '@angular/router';
import {filter, map} from 'rxjs/operators';
import {IonTabs} from '@ionic/angular/standalone';

@Component({
  selector: 'app-main',
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.scss'],
  imports: [IonTabs],
})
export class MainComponent {
  private router = inject(Router);
  isMainPage$ = this.router.events.pipe(
    filter(event => event instanceof NavigationEnd),
    map((event: NavigationEnd) => event.urlAfterRedirects === '/')
  );
}
