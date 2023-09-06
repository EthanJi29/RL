(function(){/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
'use strict';var h=this||self;function m(a,b){a=a.split(".");var c=h;a[0]in c||"undefined"==typeof c.execScript||c.execScript("var "+a[0]);for(var d;a.length&&(d=a.shift());)a.length||void 0===b?c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}:c[d]=b};var n,p;a:{for(var q=["CLOSURE_FLAGS"],w=h,x=0;x<q.length;x++)if(w=w[q[x]],null==w){p=null;break a}p=w}var y=p&&p[610401301];n=null!=y?y:!1;var z;const A=h.navigator;z=A?A.userAgentData||null:null;function B(a){return n?z?z.brands.some(({brand:b})=>b&&-1!=b.indexOf(a)):!1:!1}function C(a){var b;a:{if(b=h.navigator)if(b=b.userAgent)break a;b=""}return-1!=b.indexOf(a)};function D(){return n?!!z&&0<z.brands.length:!1}function E(){return D()?B("Chromium"):(C("Chrome")||C("CriOS"))&&!(D()?0:C("Edge"))||C("Silk")};!C("Android")||E();E();C("Safari")&&(E()||(D()?0:C("Coast"))||(D()?0:C("Opera"))||(D()?0:C("Edge"))||(D()?B("Microsoft Edge"):C("Edg/"))||D()&&B("Opera"));const F=Symbol();function G(a){const b=a[F]|0;1!==(b&1)&&(Object.isFrozen(a)&&(a=Array.prototype.slice.call(a)),a[F]=b|1)}function H(){var a=[];a[F]|=1;return a}function aa(a,b){b[F]=(a|0)&-99}function I(a,b){b[F]=(a|34)&-73;a&4&&Object.freeze(b)}function J(a){a=a>>10&1023;return 0===a?536870912:a};var K={};function L(a){return null!==a&&"object"===typeof a&&!Array.isArray(a)&&a.constructor===Object}var M;const ba=[];ba[F]=39;M=Object.freeze(ba);let N;function O(a,b){N=b;a=new a(b);N=void 0;return a};function ca(a){switch(typeof a){case "number":return isFinite(a)?a:String(a);case "boolean":return a?1:0;case "object":if(a&&!Array.isArray(a)&&null!=a&&a instanceof Uint8Array){let b="",c=0;const d=a.length-10240;for(;c<d;)b+=String.fromCharCode.apply(null,a.subarray(c,c+=10240));b+=String.fromCharCode.apply(null,c?a.subarray(c):a);return btoa(b)}}return a};function da(a,b,c){a=Array.prototype.slice.call(a);var d=a.length;const e=b&256?a[d-1]:void 0;d+=e?-1:0;for(b=b&512?1:0;b<d;b++)a[b]=c(a[b]);if(e){b=a[b]={};for(const f in e)b[f]=c(e[f])}return a}function ea(a,b,c,d,e,f){if(null!=a){if(Array.isArray(a))a=e&&0==a.length&&(a[F]|0)&1?void 0:f&&(a[F]|0)&2?a:P(a,b,c,void 0!==d,e,f);else if(L(a)){const k={};for(let g in a)k[g]=ea(a[g],b,c,d,e,f);a=k}else a=b(a,d);return a}}
function P(a,b,c,d,e,f){const k=d||c?a[F]|0:0;d=d?!!(k&32):void 0;a=Array.prototype.slice.call(a);for(let g=0;g<a.length;g++)a[g]=ea(a[g],b,c,d,e,f);c&&c(k,a);return a}function fa(a){return a.o===K?a.toJSON():ca(a)};function ja(a,b,c=I){if(null!=a){if(a instanceof Uint8Array)return b?a:new Uint8Array(a);if(Array.isArray(a)){const d=a[F]|0;return d&2?a:!b||d&68||!(d&32||0===d)?P(a,ja,d&4?I:c,!0,!1,!0):(a[F]=d|34,a)}a.o===K&&(b=a.l,c=b[F],a=c&2?a:O(a.constructor,ka(b,c,!0)));return a}}function ka(a,b,c){const d=c||b&2?I:aa,e=!!(b&32);a=da(a,b,f=>ja(f,e,d));a[F]=a[F]|32|(c?2:0);return a};function la(a,b,c,d){var e=J(b);if(c>=e){let f=b;if(b&256)e=a[a.length-1];else{if(null==d)return;e=a[e+((b>>9&1)-1)]={};f|=256}e[c]=d;f!==b&&(a[F]=f)}else a[c+((b>>9&1)-1)]=d,b&256&&(a=a[a.length-1],c in a&&delete a[c])}
function R(a,b,c){var d=a.l,e=d[F];a:if(-1===c)var f=null;else{if(c>=J(e)){if(e&256){f=d[d.length-1][c];break a}}else if(f=c+((e>>9&1)-1),f<d.length){f=d[f];break a}f=void 0}var k=!1;if(null==f||"object"!==typeof f||(k=Array.isArray(f))||f.o!==K)if(k){let g=k=f[F]|0;0===g&&(g|=e&32);g|=e&2;g!==k&&(f[F]=g);b=new b(f)}else b=void 0;else b=f;b!==f&&null!=b&&la(d,e,c,b);d=b;if(null==d)return d;a=a.l;e=a[F];e&2||(b=d,f=b.l,k=f[F],b=k&2?O(b.constructor,ka(f,k,!1)):b,b!==d&&(d=b,la(a,e,c,d)));return d};var S=class{constructor(a){a:{null==a&&(a=N);N=void 0;if(null==a){var b=96;a=[]}else{if(!Array.isArray(a))throw Error();b=a[F]|0;if(b&64)break a;var c=a;b|=64;var d=c.length;if(d){var e=d-1;d=c[e];if(L(d)){b|=256;const f=(b>>9&1)-1;e-=f;if(1024<=e){e=1023+f;const k=c.length;for(let g=e;g<k;g++){const l=c[g];null!=l&&l!==d&&(d[g-f]=l)}c.length=e+1;c[e]=d;e=1023}b=b&-1047553|(e&1023)<<10}}}a[F]=b}this.l=a}toJSON(){var a=P(this.l,fa,void 0,void 0,!1,!1);return ma(this,a,!0)}};S.prototype.o=K;
S.prototype.toString=function(){return ma(this,this.l,!1).toString()};
function ma(a,b,c){var d=a.constructor.m,e=J((c?a.l:b)[F]),f=!1;if(d){if(!c){b=Array.prototype.slice.call(b);var k;if(b.length&&L(k=b[b.length-1]))for(f=0;f<d.length;f++)if(d[f]>=e){Object.assign(b[b.length-1]={},k);break}f=!0}e=b;c=!c;k=a.l[F];a=J(k);k=(k>>9&1)-1;var g;for(let r=0;r<d.length;r++){var l=d[r];if(l<a){l+=k;var u=e[l];null==u?e[l]=c?M:H():c&&u!==M&&G(u)}else{if(!g){var v=void 0;e.length&&L(v=e[e.length-1])?g=v:e.push(g={})}u=g[l];null==g[l]?g[l]=c?M:H():c&&u!==M&&G(u)}}}d=b.length;if(!d)return b;
let ha,ia;if(L(g=b[d-1])){a:{var t=g;v={};e=!1;for(let r in t)c=t[r],Array.isArray(c)&&c!=c&&(e=!0),null!=c?v[r]=c:e=!0;if(e){for(let r in v){t=v;break a}t=null}}t!=g&&(ha=!0);d--}for(;0<d;d--){g=b[d-1];if(null!=g)break;ia=!0}if(!ha&&!ia)return b;var Q;f?Q=b:Q=Array.prototype.slice.call(b,0,d);b=Q;f&&(b.length=d);t&&b.push(t);return b};function na(a){return b=>{if(null==b||""==b)b=new a;else{b=JSON.parse(b);if(!Array.isArray(b))throw Error(void 0);b[F]|=32;b=O(a,b)}return b}};var T=class extends S{};T.m=[17];var U=class extends S{};U.m=[27];var V=class extends S{};V.m=[8];var oa=na(class extends S{});var pa=class extends S{},qa=na(pa);pa.m=[1,2,3];function W(a,b){a=a.getElementsByTagName("META");for(let c=0;c<a.length;++c)if(a[c].getAttribute("name")===b)return a[c].getAttribute("content")||"";return""};function X(a,b){a=a.body;var c={detail:void 0};let d;"function"===typeof window.CustomEvent?d=new CustomEvent(b,c):(d=document.createEvent("CustomEvent"),d.initCustomEvent(b,!!c.bubbles,!!c.cancelable,c.detail));a.dispatchEvent(d)}
var ra=class{constructor(a){this.body=a;this.g=[];W(a,"sampling_mod");var b=W(a,"namespace");if(!b){b="ns-"+(0,Math.random)().toString(36).substr(2,5);a:{var c=a.getElementsByTagName("META");for(let d=0;d<c.length;++d)if("namespace"===c[d].getAttribute("name")&&c[d].getAttribute("index")===(0).toString()){c[d].setAttribute("content",b);break a}c=a.querySelector("#mys-meta");c||(c=document.createElement("div"),c.id="mys-meta",c.style.position="absolute",c.style.display="none",a.appendChild(c));a=document.createElement("META");
a.setAttribute("name","namespace");a.setAttribute("content",b);a.setAttribute("index",(0).toString());c.appendChild(a)}}if(!(0<this.g.length)){a=W(this.body,"environment");for(const d of a.split("|"))d&&this.g.push(d)}}addEventListener(a,b){this.body.addEventListener(a,b)}};function sa(a){var b=document;a=String(a);"application/xhtml+xml"===b.contentType&&(a=a.toLowerCase());return b.createElement(a)};function ta(a){if(1>=a.i.offsetWidth||1>=a.i.offsetHeight)return!1;a.g.remove();X(a.context,"spanReady");return!0}
var ua=class{constructor(a){this.context=a;this.j={A:!1,v:100};this.i=sa("SPAN");this.g=sa("DIV");this.i.style.fontSize="6px";this.i.textContent="go";this.g.style.position="absolute";this.g.style.top="100%";this.g.style.left="100%";this.g.style.width="1px";this.g.style.height="0";this.g.style.overflow="hidden";this.g.style.visibility="hidden";this.g.appendChild(this.i)}wait(){if(!this.j.A&&(X(this.context,"spanStart"),this.context.body.appendChild(this.g),!ta(this)))return new Promise(a=>{const b=
setInterval(()=>{ta(this)&&(clearInterval(b),a())},this.j.v)})}};var va=class{constructor(a,b){this.context=a;this.g=R(b,U,1)||new U;R(b,V,12)||new V;R(this.g,T,10)||new T}};function wa(a){a.j.length=0;a.i=!0}function xa(a,b){a.g=!0;const c=()=>{a.i=!1;const d=a.j.shift();return void 0===d?(a.g=!1,Promise.resolve()):xa(a,d())};return b?b.then(c,()=>{if(a.i)return c();a.g=!1;return Promise.reject()}):c()}function ya(a,b){for(const c of b)a.j.push(c);if(!a.g)return xa(a)}var za=class{constructor(){this.i=this.g=!1;this.j=[]}};function Aa(a){wa(a.j);return ya(a.j,[()=>{if(!a.s){var b=W(a.context.body,"render_config")||"[]";b=oa(b);b=new va(a.context,b);a.s=b}b=(new ua(a.context)).wait();X(a.context,"browserStart");X(a.context,"browserStartEnd");a.g&=-31;a.g|=2;return b},()=>{X(a.context,"browserReady");X(a.context,"browserReadyEnd");a.g|=4;X(a.context,"overallReady")},()=>{X(a.context,"browserQuiet");X(a.context,"browserQuietEnd");a.g|=8}])}
function Ba(a){qa(W(a.context.body,"engine_msg")||"[]");return Aa(a)||Promise.resolve()}var Y=class{constructor(a,b){this.j=new za;this.g=0;this.context=new ra(b)}u(){return this.g}i(){this.g&=-31;this.g|=1;let a=0;const b=this.context.body;b.addEventListener("browserRender",()=>{++a;if(1===a)X(this.context,"overallStart"),Ba(this).then(()=>{X(this.context,"overallQuiet")});else{var c=b.clientHeight;b.clientWidth&&c&&Ba(this)}})}};let Z;m("mys.engine.init",(a,b)=>{Z=new Y(a,b);Z.i()});m("mys.engine.stage",()=>{let a;return(null==(a=Z)?void 0:a.g)||0});m("mys.Engine",Y);m("mys.Engine.prototype.i",Y.prototype.i);m("mys.Engine.prototype.s",Y.prototype.u);}).call(this);
