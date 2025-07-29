# 🎯 Auto Hand Selection System - Soluzione al High Card Spam

## 🔍 Problema Identificato

Il modello RL spammava **64.9% high_card** e solo **30.8% pair**, non riuscendo a imparare combinazioni migliori nonostante avesse 8 carte in mano e la possibilità di giocare fino a 5 carte.

### Cause del Problema

1. **Action Space Complesso**: Il modello doveva selezionare manualmente ogni carta (2^8 = 256 azioni possibili)
2. **Selezione Subottimale**: Il modello spesso selezionava 1-2 carte casuali → high_card garantito
3. **Mancanza di Ottimizzazione Automatica**: L'ambiente non cercava la migliore combinazione possibile

## 🚀 Soluzione Implementata

### Modifiche Principali

#### 1. Funzione `_find_best_5_card_hand()`
```python
def _find_best_5_card_hand(self, cards: List[Card]) -> Tuple[List[Card], HandType, int, int]:
    """Find the best possible 5-card hand from the given cards"""
    if len(cards) < 5:
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        return cards, hand_type, chips, mult
    
    # Try all combinations of 5 cards and find the best one
    from itertools import combinations
    best_hand = None
    best_type = HandType.HIGH_CARD
    best_chips = 0
    best_mult = 0
    best_score = 0
    
    for combo in combinations(cards, 5):
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(list(combo))
        score = chips * mult
        
        # Hand type hierarchy for comparison
        type_values = {
            HandType.HIGH_CARD: 1,
            HandType.PAIR: 2,
            HandType.TWO_PAIR: 3,
            HandType.THREE_OF_A_KIND: 4,
            HandType.STRAIGHT: 5,
            HandType.FLUSH: 6,
            HandType.FULL_HOUSE: 7,
            HandType.FOUR_OF_A_KIND: 8,
            HandType.STRAIGHT_FLUSH: 9,
            HandType.ROYAL_FLUSH: 10
        }
        
        type_value = type_values.get(hand_type, 0)
        
        # Compare by hand type first, then by score
        if (type_value > type_values.get(best_type, 0)) or \
           (type_value == type_values.get(best_type, 0) and score > best_score):
            best_hand = list(combo)
            best_type = hand_type
            best_chips = chips
            best_mult = mult
            best_score = score
    
    return best_hand, best_type, best_chips, best_mult
```

#### 2. Modifica di `_execute_play()`
```python
def _execute_play(self, selected_cards: List[Card], info: Dict) -> Tuple[float, Dict]:
    # ... validazioni precedenti ...
    
    # 🚀 NUOVO: Trova automaticamente la migliore combinazione di 5 carte!
    if len(self.hand) >= 5:
        best_cards, hand_type, chips, mult = self._find_best_5_card_hand(self.hand)
        actual_played_cards = best_cards
        info['cards_analyzed'] = len(self.hand)
        info['best_hand_found'] = hand_type.value
        info['original_selection'] = len(selected_cards)
    else:
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(self.hand)
        actual_played_cards = self.hand.copy()
        info['cards_analyzed'] = len(self.hand)
        info['best_hand_found'] = hand_type.value
        info['original_selection'] = len(selected_cards)
    
    # ... resto della funzione usa actual_played_cards invece di selected_cards ...
```

#### 3. Aggiornamento dei Reward
- Tutti i reward ora si basano su `actual_played_cards` (migliore combinazione)
- Bonus per carta giocata basato sulle carte effettivamente utilizzate
- Exploration bonus allineato alla strategia ottimale

#### 4. Info Dettagliate
```python
'auto_hand_analysis': {
    'total_cards_in_hand': len(self.hand) + len(actual_played_cards),
    'cards_played': len(actual_played_cards),
    'best_hand_found': hand_type.value,
    'original_user_selection': len(selected_cards),
    'improvement_found': hand_type != HandType.HIGH_CARD or len(actual_played_cards) > 1
}
```

## 📊 Risultati Attesi

### Distribuzione Mani - Prima vs Dopo

| Hand Type          | Prima % | Dopo % | Cambiamento |
|-------------------|---------|--------|-------------|
| high_card         | 64.9    | 15.0   | -49.9%      |
| pair              | 30.8    | 45.0   | +14.2%      |
| two_pair          | 2.4     | 15.0   | +12.6%      |
| three_of_a_kind   | 1.6     | 12.0   | +10.4%      |
| straight          | 0.1     | 4.0    | +3.9%       |
| flush             | 0.1     | 4.0    | +3.9%       |
| full_house        | 0.1     | 3.0    | +2.9%       |
| four_of_a_kind    | 0.0     | 2.0    | +2.0%       |

### Impatto sui Punteggi

1. **Riduzione Penalità**: High card da 64.9% a 15.0% = meno penalità (-2.0 reward)
2. **Aumento Bonus**: Combinazioni migliori = reward molto più alti
3. **Punteggi Mano**: Chips × Mult molto superiori grazie a combinazioni migliori
4. **Progressione Ante**: Più efficace grazie a punteggi più alti

## 🧪 Testing

### Test Files Creati

1. **`test_auto_hand_selection.py`**: Test unitari per verificare la funzione
2. **`quick_test_auto_hands.py`**: Training rapido (50k timesteps) per validare

### Comandi di Test

```bash
# Test unitari
python test_auto_hand_selection.py

# Test rapido con training
python quick_test_auto_hands.py

# Training completo con curriculum
python src/training.py
```

### Metriche di Successo

- ✅ **High card < 20%** (vs 64.9% precedente)
- ✅ **Pair + Two_pair + Three_of_a_kind > 60%** (vs 34.8% precedente)
- ✅ **Auto-improvement rate > 80%** (nuova metrica)

## 🔧 Vantaggi del Sistema

### 1. **Semplicità per il Modello**
- Non deve più imparare selezioni complesse di carte
- Focus su strategia high-level (play vs discard vs shop)
- Riduzione drastica dello spazio di azione effettivo

### 2. **Ottimalità Garantita**
- Sempre gioca la migliore combinazione possibile
- Elimina errori di selezione subottimale
- Massimizza il potenziale di ogni mano

### 3. **Apprendimento Accelerato**
- Reward più coerenti e prevedibili
- Riduzione del rumore nell'apprendimento
- Focus su strategie di alto livello

### 4. **Compatibilità**
- Mantiene l'action space esistente
- Non richiede modifiche al modello PPO
- Retrocompatibile con modelli esistenti

## 🎯 Prossimi Passi

1. **Validazione**: Eseguire `quick_test_auto_hands.py` per verifica rapida
2. **Training Completo**: Curriculum learning con le nuove modifiche
3. **Monitoraggio**: Verificare distribuzione mani durante training
4. **Ottimizzazione**: Eventuale fine-tuning dei reward se necessario

## 📈 Aspettative di Performance

Con questa modifica, ci aspettiamo:
- **Reward medio**: Da ~10-50 a ~100-300 per episodio
- **Ante progression**: Più consistente verso ante 4-8
- **Win rate**: Miglioramento significativo
- **Training stability**: Molto più stabile e prevedibile

---

*Implementato il 27 Gennaio 2025 per risolvere il problema del high_card spam nel training RL di Balatro.*
