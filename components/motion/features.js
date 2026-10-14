/* Motion's animation features, loaded on their own after first paint — see
   LazyMotion in components/providers.js. domMax rather than domAnimation
   because the sample reading's toggle slides its highlight with `layoutId`. */
import { domMax } from 'framer-motion';

export default domMax;
