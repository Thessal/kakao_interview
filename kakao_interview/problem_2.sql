
SELECT user_id
FROM (
         SELECT user_id,
                transaction_id,
                sum((CASE WHEN payment_action_type = 'PAYMENT' THEN 1 ELSE -1 END) * amount) AS net_amount
         FROM (
                  SELECT distinct users.user_id, transaction_id, transacted_at, payment_action_type, amount
                  FROM users
                           INNER JOIN dutchpay_claim dc ON (users.user_id = dc.claim_user_id)
                           INNER JOIN a_payment_trx dc ON (users.user_id = dc.user_id)
              )
         WHERE (
                   transacted_at >= '2019-12-01 00:00:00'
                   AND
                   transacted_at < (
                       CASE
                           WHEN payment_action_type = 'PAYMENT' THEN '2020-01-01 00:00:00'
                           ELSE '2020-03-01 00:00:00'
                           END
                       )
               )
         GROUP BY transaction_id
         HAVING net_amount > 0
     )
GROUP BY user_id
HAVING sum(net_amount) >= 10000
